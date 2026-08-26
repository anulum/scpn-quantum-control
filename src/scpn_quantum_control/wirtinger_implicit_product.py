# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Wirtinger + implicit differentiation product
"""Fail-closed **Wirtinger + implicit differentiation** product surface.

Productises complex-valued Wirtinger contracts and real-valued implicit
sensitivity helpers as a first-class product: versioned surface catalogue,
materialised scalar demo probes, and refuse-complex-without-Wirtinger paths
composing unsuitable scenario
``unsuitable:complex.objective_without_wirtinger`` with metamorphic law
``law:anti_silent.complex_without_wirtinger``.

Composes ambient :mod:`scpn_quantum_control.wirtinger_calculus` and
:mod:`scpn_quantum_control.differentiable_implicit_sensitivity` — does **not**
re-architect engines or invent-green full holomorphic QFT AD or planner-matrix
rows; both remain explicit product boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

SurfaceKind = Literal[
    "wirtinger_partials",
    "holomorphic_gradient",
    "real_objective_cr_gradient",
    "implicit_stationary",
    "implicit_fixed_point",
    "complex_without_contract_refuse",
]
"""Catalogue kinds for Wirtinger / implicit product rows."""

SupportPosture = Literal[
    "local_materialised",
    "policy_only",
    "refuse_only",
]
"""Support posture badges for product rows."""

WIRTINGER_IMPLICIT_PRODUCT_SCHEMA: Final[str] = "wirtinger_implicit_product.v1"
"""JSON schema identifier for serialised product payloads."""

WIRTINGER_IMPLICIT_CLAIM_BOUNDARY: Final[str] = (
    "Wirtinger + implicit differentiation product surface only; catalogues "
    "Wirtinger partials / holomorphic / CR real-objective gradients and "
    "implicit stationary/fixed-point sensitivity; materialised local scalar "
    "demos only; composes the complex-without-Wirtinger unsuitable scenario and "
    "anti-silent metamorphic law; does not invent-green full holomorphic QFT "
    "AD, planner matrix rows, or hardware gradients"
)
"""Shared claim boundary for Wirtinger / implicit product payloads."""

COMPLEX_OBJECTIVE_WITHOUT_WIRTINGER_SCENARIO: Final[str] = (
    "unsuitable:complex.objective_without_wirtinger"
)
"""Unsuitable-scenario id for a complex objective without Wirtinger."""

COMPLEX_OBJECTIVE_WIRTINGER_LAW: Final[str] = "law:anti_silent.complex_without_wirtinger"
"""Metamorphic-law id paired with the complex-objective refusal."""


@dataclass(frozen=True, slots=True)
class WirtingerImplicitSurfaceRow:
    """One product catalogue row for a Wirtinger or implicit surface.

    Attributes
    ----------
    surface_id
        Stable catalogue identifier.
    kind
        Surface kind.
    title
        Human-readable title.
    summary
        Short description.
    module_path
        Primary ambient module path.
    symbol_name
        Primary ambient symbol.
    support_posture
        Support posture badge.
    unsuitable_scenario_pointer
        Unsuitable-scenario pointer required for complex paths.
    metamorphic_verification_pointer
        Anti-silent complex metamorphic-law pointer.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    surface_id: str
    kind: SurfaceKind
    title: str
    summary: str
    module_path: str
    symbol_name: str
    support_posture: SupportPosture
    unsuitable_scenario_pointer: str = COMPLEX_OBJECTIVE_WITHOUT_WIRTINGER_SCENARIO
    metamorphic_verification_pointer: str = COMPLEX_OBJECTIVE_WIRTINGER_LAW
    as_of: str = "2026-07-24"
    claim_boundary: str = WIRTINGER_IMPLICIT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate catalogue row invariants."""
        if not self.surface_id or not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if self.kind not in {
            "wirtinger_partials",
            "holomorphic_gradient",
            "real_objective_cr_gradient",
            "implicit_stationary",
            "implicit_fixed_point",
            "complex_without_contract_refuse",
        }:
            raise ValueError(f"unknown surface kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.symbol_name or not self.symbol_name.strip():
            raise ValueError("symbol_name must be non-empty")
        if self.support_posture not in {
            "local_materialised",
            "policy_only",
            "refuse_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if not self.unsuitable_scenario_pointer or not self.unsuitable_scenario_pointer.strip():
            raise ValueError("unsuitable_scenario_pointer must be non-empty")
        if (
            not self.metamorphic_verification_pointer
            or not self.metamorphic_verification_pointer.strip()
        ):
            raise ValueError("metamorphic_verification_pointer must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "surface_id": self.surface_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "support_posture": self.support_posture,
            "unsuitable_scenario_pointer": self.unsuitable_scenario_pointer,
            "metamorphic_verification_pointer": self.metamorphic_verification_pointer,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedWirtingerProbe:
    """Materialised local Wirtinger probe result.

    Attributes
    ----------
    z
        Evaluation point (complex components as real/imag pairs).
    df_dz
        Wirtinger ``df/dz`` components as (real, imag) pairs.
    df_dconj_z
        Wirtinger ``df/dconj_z`` components as (real, imag) pairs.
    holomorphic_residual
        ``max|df/dconj_z|`` residual.
    is_holomorphic
        Whether residual is at or below tolerance.
    demo_label
        Which demo objective was materialised.

    """

    z: tuple[tuple[float, float], ...]
    df_dz: tuple[tuple[float, float], ...]
    df_dconj_z: tuple[tuple[float, float], ...]
    holomorphic_residual: float
    is_holomorphic: bool
    demo_label: str
    claim_boundary: str = WIRTINGER_IMPLICIT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised Wirtinger probe invariants."""
        if not self.z:
            raise ValueError("z must be non-empty")
        if len(self.df_dz) != len(self.z):
            raise ValueError("df_dz length must match z")
        if len(self.df_dconj_z) != len(self.z):
            raise ValueError("df_dconj_z length must match z")
        if self.holomorphic_residual < 0.0:
            raise ValueError("holomorphic_residual must be non-negative")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "z": [list(pair) for pair in self.z],
            "df_dz": [list(pair) for pair in self.df_dz],
            "df_dconj_z": [list(pair) for pair in self.df_dconj_z],
            "holomorphic_residual": self.holomorphic_residual,
            "is_holomorphic": self.is_holomorphic,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedImplicitProbe:
    """Materialised local implicit-sensitivity probe.

    Attributes
    ----------
    method
        Ambient method name.
    sensitivity
        Row-major flattened sensitivity matrix.
    shape
        ``(n_params, n_hyper)`` shape.
    condition_number
        Reported condition number.
    demo_label
        Which demo was materialised.

    """

    method: str
    sensitivity: tuple[float, ...]
    shape: tuple[int, int]
    condition_number: float
    demo_label: str
    claim_boundary: str = WIRTINGER_IMPLICIT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised implicit probe invariants."""
        if not self.method or not self.method.strip():
            raise ValueError("method must be non-empty")
        if not self.sensitivity:
            raise ValueError("sensitivity must be non-empty")
        rows, cols = self.shape
        if rows <= 0 or cols <= 0:
            raise ValueError("shape dimensions must be positive")
        if len(self.sensitivity) != rows * cols:
            raise ValueError("sensitivity length must equal rows*cols")
        if not np.isfinite(self.condition_number) or self.condition_number < 0.0:
            raise ValueError("condition_number must be finite and non-negative")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "method": self.method,
            "sensitivity": list(self.sensitivity),
            "shape": list(self.shape),
            "condition_number": self.condition_number,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ComplexContractDecision:
    """Fail-closed decision for complex objectives without Wirtinger.

    Attributes
    ----------
    allowed
        Always False for the refuse path (silent real-grad substitution forbidden).
    has_wirtinger_contract
        Whether the caller declared a Wirtinger contract.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    scenario_id
        Unsuitable-scenario id.
    metamorphic_law_id
        Anti-silent metamorphic-law id.

    """

    allowed: bool
    has_wirtinger_contract: bool
    reason: str
    blockers: tuple[str, ...]
    scenario_id: str = COMPLEX_OBJECTIVE_WITHOUT_WIRTINGER_SCENARIO
    metamorphic_law_id: str = COMPLEX_OBJECTIVE_WIRTINGER_LAW
    claim_boundary: str = WIRTINGER_IMPLICIT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate complex-contract decision invariants."""
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and not self.has_wirtinger_contract:
            raise ValueError("allowed requires has_wirtinger_contract=True")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "allowed": self.allowed,
            "has_wirtinger_contract": self.has_wirtinger_contract,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "scenario_id": self.scenario_id,
            "metamorphic_law_id": self.metamorphic_law_id,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    surface_id: str,
    *,
    kind: SurfaceKind,
    title: str,
    summary: str,
    module_path: str,
    symbol_name: str,
    support_posture: SupportPosture,
) -> WirtingerImplicitSurfaceRow:
    """Build one catalogue row."""
    return WirtingerImplicitSurfaceRow(
        surface_id=surface_id,
        kind=kind,
        title=title,
        summary=summary,
        module_path=module_path,
        symbol_name=symbol_name,
        support_posture=support_posture,
    )


_CANONICAL_SURFACES: Final[tuple[WirtingerImplicitSurfaceRow, ...]] = (
    _row(
        "wirtinger_partials",
        kind="wirtinger_partials",
        title="Wirtinger partials df/dz and df/dconj_z",
        summary=(
            "Central-difference Wirtinger partials for complex callables with "
            "holomorphicity residual (Cauchy-Riemann residual)."
        ),
        module_path="scpn_quantum_control.wirtinger_calculus",
        symbol_name="wirtinger_partials",
        support_posture="local_materialised",
    ),
    _row(
        "holomorphic_gradient",
        kind="holomorphic_gradient",
        title="Holomorphic complex derivative df/dz",
        summary=(
            "Fail-closed ordinary complex derivative when Cauchy-Riemann holds; "
            "raises when residual exceeds tolerance."
        ),
        module_path="scpn_quantum_control.wirtinger_calculus",
        symbol_name="holomorphic_gradient",
        support_posture="local_materialised",
    ),
    _row(
        "real_objective_cr_gradient",
        kind="real_objective_cr_gradient",
        title="CR steepest-descent gradient for real losses",
        summary=(
            "Conjugate Wirtinger gradient dL/dconj_z for real-valued losses on "
            "complex parameters (CR steepest descent)."
        ),
        module_path="scpn_quantum_control.wirtinger_calculus",
        symbol_name="real_objective_gradient",
        support_posture="local_materialised",
    ),
    _row(
        "implicit_stationary_sensitivity",
        kind="implicit_stationary",
        title="Implicit stationary sensitivity dx*/dalpha",
        summary=(
            "Solve dx*/dalpha = -H^-1 B for stationary optima with symmetric "
            "positive-definite Hessian and cross derivatives."
        ),
        module_path="scpn_quantum_control.differentiable_implicit_sensitivity",
        symbol_name="implicit_stationary_sensitivity",
        support_posture="local_materialised",
    ),
    _row(
        "implicit_fixed_point_sensitivity",
        kind="implicit_fixed_point",
        title="Implicit fixed-point sensitivity",
        summary=(
            "Solve (I - dT/dx)^-1 dT/dalpha for fixed points x* = T(x*, alpha) "
            "with condition-number guards."
        ),
        module_path="scpn_quantum_control.differentiable_implicit_sensitivity",
        symbol_name="implicit_fixed_point_sensitivity",
        support_posture="local_materialised",
    ),
    _row(
        "complex_without_wirtinger_refuse",
        kind="complex_without_contract_refuse",
        title="Refuse complex objective without Wirtinger contract",
        summary=(
            "Product refuse path composing the unsuitable scenario and anti-silent "
            "metamorphic law; forbids silent real-gradient substitution."
        ),
        module_path="scpn_quantum_control.unsuitable_scenario_registry",
        symbol_name="unsuitable:complex.objective_without_wirtinger",
        support_posture="refuse_only",
    ),
)


def _catalogue_map() -> dict[str, WirtingerImplicitSurfaceRow]:
    """Return surface_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, WirtingerImplicitSurfaceRow] = {}
    for row in _CANONICAL_SURFACES:
        key = row.surface_id.strip()
        if not key:
            raise RuntimeError("wirtinger/implicit catalogue contains blank surface_id")
        if key in mapping:
            raise RuntimeError(f"duplicate surface_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("wirtinger/implicit catalogue must be non-empty")
    return mapping


_SURFACE_BY_ID: Final[Mapping[str, WirtingerImplicitSurfaceRow]] = _catalogue_map()


def list_wirtinger_implicit_surface_ids() -> tuple[str, ...]:
    """Return all product surface identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered surface identifiers.

    """
    return tuple(row.surface_id for row in _CANONICAL_SURFACES)


def get_wirtinger_implicit_surface(surface_id: str) -> WirtingerImplicitSurfaceRow:
    """Return one surface row or raise for blank/unknown identifiers.

    Parameters
    ----------
    surface_id
        Catalogue surface key.

    Returns
    -------
    WirtingerImplicitSurfaceRow
        Matching row.

    Raises
    ------
    ValueError
        If ``surface_id`` is blank or unknown (fail closed).

    """
    if not surface_id or not str(surface_id).strip():
        raise ValueError("surface_id must be a non-empty string")
    key = str(surface_id).strip()
    try:
        return _SURFACE_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown surface_id {key!r}; refuse invent-green Wirtinger/implicit "
            f"product claim (known_count={len(_SURFACE_BY_ID)})"
        ) from exc


def iter_wirtinger_implicit_surfaces(
    *,
    kind: SurfaceKind | None = None,
    support_posture: SupportPosture | None = None,
) -> tuple[WirtingerImplicitSurfaceRow, ...]:
    """Return filtered surface rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[WirtingerImplicitSurfaceRow, ...]
        Matching rows.

    """
    rows: Sequence[WirtingerImplicitSurfaceRow] = _CANONICAL_SURFACES
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def decide_complex_objective_contract(
    *,
    has_wirtinger_contract: bool,
) -> ComplexContractDecision:
    """Decide whether a complex objective path may proceed.

    Parameters
    ----------
    has_wirtinger_contract
        Whether the caller declared an explicit Wirtinger contract.

    Returns
    -------
    ComplexContractDecision
        Allowed only when a Wirtinger contract is declared; otherwise refuse
        with unsuitable-scenario and metamorphic-law pointers; silent
        real-gradient substitution is forbidden.

    """
    if has_wirtinger_contract:
        return ComplexContractDecision(
            allowed=True,
            has_wirtinger_contract=True,
            reason=(
                "complex objective allowed under explicit Wirtinger contract "
                f"(compose {COMPLEX_OBJECTIVE_WITHOUT_WIRTINGER_SCENARIO} + "
                f"{COMPLEX_OBJECTIVE_WIRTINGER_LAW})"
            ),
            blockers=(),
        )
    return ComplexContractDecision(
        allowed=False,
        has_wirtinger_contract=False,
        reason=(
            "complex objective refused without Wirtinger contract; silent "
            "real-gradient substitution is forbidden"
        ),
        blockers=(
            f"{COMPLEX_OBJECTIVE_WITHOUT_WIRTINGER_SCENARIO}: complex-valued objective "
            "without explicit Wirtinger contract",
            f"{COMPLEX_OBJECTIVE_WIRTINGER_LAW}: anti-silent complex without Wirtinger",
        ),
    )


def materialise_demo_wirtinger_probe(
    *,
    demo: Literal["holomorphic_square", "modulus_squared"] = "holomorphic_square",
    z0: complex = 1.0 + 0.5j,
    step: float = 1e-6,
    tolerance: float = 1e-6,
) -> MaterialisedWirtingerProbe:
    """Run a deterministic local Wirtinger probe on a scalar demo.

    Demos:

    * ``holomorphic_square`` — ``f(z) = z^2`` (holomorphic; ``df/dz = 2z``).
    * ``modulus_squared`` — ``f(z) = |z|^2`` (non-holomorphic; CR residual > 0).

    Parameters
    ----------
    demo
        Demo objective label.
    z0
        Evaluation point.
    step
        Central-difference step for ambient Wirtinger partials.
    tolerance
        Holomorphicity residual threshold.

    Returns
    -------
    MaterialisedWirtingerProbe
        Partials and residual with non-empty primary observables.

    Raises
    ------
    ValueError
        If ``demo`` is unknown or ambient validation fails.

    """
    from .wirtinger_calculus import wirtinger_partials

    if demo == "holomorphic_square":

        def objective(vector: NDArray[np.complex128]) -> complex:
            return complex(vector[0] * vector[0])

        label = "holomorphic_square"
    elif demo == "modulus_squared":

        def objective(vector: NDArray[np.complex128]) -> complex:
            return complex(float(np.abs(vector[0]) ** 2), 0.0)

        label = "modulus_squared"
    else:
        raise ValueError(
            f"unknown wirtinger demo {demo!r}; known: holomorphic_square, modulus_squared"
        )

    z = np.asarray([z0], dtype=np.complex128)
    derivative = wirtinger_partials(objective, z, step=step)
    residual = float(derivative.holomorphic_residual)
    return MaterialisedWirtingerProbe(
        z=((float(z0.real), float(z0.imag)),),
        df_dz=((float(derivative.df_dz[0].real), float(derivative.df_dz[0].imag)),),
        df_dconj_z=((float(derivative.df_dconj_z[0].real), float(derivative.df_dconj_z[0].imag)),),
        holomorphic_residual=residual,
        is_holomorphic=residual <= tolerance,
        demo_label=label,
    )


def materialise_demo_implicit_stationary_probe(
    *,
    hessian_scale: float = 2.0,
    cross_scale: float = 1.0,
) -> MaterialisedImplicitProbe:
    """Materialise a one-dimensional stationary-sensitivity demo.

    Uses ``H = [[hessian_scale]]`` and ``B = [[cross_scale]]`` so
    ``dx*/dalpha = -cross_scale / hessian_scale`` when ``hessian_scale > 0``.

    Parameters
    ----------
    hessian_scale
        Positive diagonal Hessian entry.
    cross_scale
        Cross-derivative entry.

    Returns
    -------
    MaterialisedImplicitProbe
        Sensitivity vector and condition metadata.

    Raises
    ------
    ValueError
        If ambient validation fails (e.g. non-positive Hessian).

    """
    from .differentiable_implicit_sensitivity import implicit_stationary_sensitivity

    if not np.isfinite(hessian_scale) or hessian_scale <= 0.0:
        raise ValueError("hessian_scale must be a positive finite value")
    if not np.isfinite(cross_scale):
        raise ValueError("cross_scale must be finite")

    result = implicit_stationary_sensitivity(
        [[float(hessian_scale)]],
        [[float(cross_scale)]],
        hyperparameter_names=("alpha0",),
    )
    sens = np.asarray(result.sensitivity, dtype=np.float64)
    flat = tuple(float(v) for v in sens.ravel())
    if not flat:
        raise ValueError("implicit probe returned empty sensitivity")
    return MaterialisedImplicitProbe(
        method=str(result.method),
        sensitivity=flat,
        shape=(int(sens.shape[0]), int(sens.shape[1])),
        condition_number=float(result.condition_number),
        demo_label="stationary_1d_scale",
    )


def map_wirtinger_implicit_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of Wirtinger / implicit product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for surface in _CANONICAL_SURFACES:
        path = surface.module_path
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "module_path": path,
                "role": "wirtinger_implicit_product_surface",
                "support_posture": surface.support_posture,
                "surface_ids": [
                    s.surface_id for s in _CANONICAL_SURFACES if s.module_path == path
                ],
                "unsuitable_scenario_pointer": surface.unsuitable_scenario_pointer,
                "metamorphic_verification_pointer": surface.metamorphic_verification_pointer,
                "claim_boundary": WIRTINGER_IMPLICIT_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_wirtinger_implicit_product_registry() -> dict[str, object]:
    """Build the full serialisable Wirtinger / implicit product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with surfaces (no blanks).

    """
    surfaces = [row.to_dict() for row in _CANONICAL_SURFACES]
    return {
        "schema": WIRTINGER_IMPLICIT_PRODUCT_SCHEMA,
        "claim_boundary": WIRTINGER_IMPLICIT_CLAIM_BOUNDARY,
        "surface_count": len(surfaces),
        "blank_entry_count": 0,
        "default_surface_id": "wirtinger_partials",
        "complex_objective_without_wirtinger": COMPLEX_OBJECTIVE_WITHOUT_WIRTINGER_SCENARIO,
        "complex_objective_wirtinger_law": COMPLEX_OBJECTIVE_WIRTINGER_LAW,
        "public_surfaces": list(map_wirtinger_implicit_public_surfaces()),
        "surfaces": surfaces,
        "policy_note": (
            "Wirtinger + implicit product catalogue only; ambient "
            "wirtinger_calculus / differentiable_implicit_sensitivity remain the "
            "implementation; full metamorphic expansion and planner-matrix rows "
            "remain open boundaries; no invent-green holomorphic QFT AD."
        ),
    }


def assert_wirtinger_implicit_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers surfaces without blanks or invent-green.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_wirtinger_implicit_product_registry`.

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
        dict(payload) if payload is not None else build_wirtinger_implicit_product_registry()
    )
    surfaces = registry.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError(
            "wirtinger/implicit product registry must contain a non-empty surfaces list"
        )
    seen: set[str] = set()
    blank = 0
    default_found = False
    refuse_found = False
    for index, row in enumerate(surfaces):
        if not isinstance(row, Mapping):
            raise ValueError(f"surface row {index} must be a mapping")
        surface_id = row.get("surface_id")
        kind = row.get("kind")
        symbol_name = row.get("symbol_name")
        scenario_row = row.get("unsuitable_scenario_pointer")
        if not surface_id or not str(surface_id).strip():
            blank += 1
            continue
        sid = str(surface_id).strip()
        if sid in seen:
            raise ValueError(f"duplicate surface_id in registry: {sid!r}")
        seen.add(sid)
        if sid == "wirtinger_partials":
            default_found = True
        if sid == "complex_without_wirtinger_refuse":
            refuse_found = True
        if kind not in {
            "wirtinger_partials",
            "holomorphic_gradient",
            "real_objective_cr_gradient",
            "implicit_stationary",
            "implicit_fixed_point",
            "complex_without_contract_refuse",
        }:
            blank += 1
            continue
        if not symbol_name or not str(symbol_name).strip():
            raise ValueError(f"surface {sid!r} must have symbol_name")
        if not scenario_row or not str(scenario_row).strip():
            raise ValueError(f"surface {sid!r} must have unsuitable_scenario_pointer")
    if blank:
        raise ValueError(
            f"wirtinger/implicit product registry has {blank} blank or invalid entries"
        )
    if not default_found:
        raise ValueError("wirtinger/implicit product registry missing wirtinger_partials")
    if not refuse_found:
        raise ValueError(
            "wirtinger/implicit product registry missing complex_without_wirtinger_refuse"
        )
    expected = set(list_wirtinger_implicit_surface_ids())
    if seen != expected:
        raise ValueError(
            f"registry surface set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    surface_count = registry.get("surface_count", -1)
    if not isinstance(surface_count, int) or surface_count != len(surfaces):
        raise ValueError("surface_count does not match surfaces list length")
    return registry


__all__ = [
    "COMPLEX_OBJECTIVE_WIRTINGER_LAW",
    "COMPLEX_OBJECTIVE_WITHOUT_WIRTINGER_SCENARIO",
    "WIRTINGER_IMPLICIT_CLAIM_BOUNDARY",
    "WIRTINGER_IMPLICIT_PRODUCT_SCHEMA",
    "ComplexContractDecision",
    "MaterialisedImplicitProbe",
    "MaterialisedWirtingerProbe",
    "SupportPosture",
    "SurfaceKind",
    "WirtingerImplicitSurfaceRow",
    "assert_wirtinger_implicit_product_integrity",
    "build_wirtinger_implicit_product_registry",
    "decide_complex_objective_contract",
    "get_wirtinger_implicit_surface",
    "iter_wirtinger_implicit_surfaces",
    "list_wirtinger_implicit_surface_ids",
    "map_wirtinger_implicit_public_surfaces",
    "materialise_demo_implicit_stationary_probe",
    "materialise_demo_wirtinger_probe",
]
