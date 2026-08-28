# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — formal + metamorphic AD verification catalogue
"""Versioned metamorphic AD verification catalogue and pure residual checks.

This product surface defends gradient correctness **beyond one-off examples**: a
catalogue of metamorphic laws, fail-closed refuse paths (including invent-green
hardware formal-proof claims), and pure residual band checks that unit tests can
drive without hardware or optional frameworks.

It composes existing transform-algebra / program-AD evidence via
``evidence_modules`` pointers. It does **not** claim interactive theorem proofs of
quantum mechanics or formal verification of hardware.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

LawKind = Literal[
    "metamorphic_identity",
    "fd_agreement_band",
    "anti_silent_wrong",
    "formal_boundary",
]
"""Classification of one verification catalogue entry."""

LawOutcome = Literal[
    "executable_local",
    "evidence_gated",
    "permanent_boundary",
    "refuse_invent_green",
]
"""Expected governance outcome for a law entry."""

METAMORPHIC_AD_VERIFICATION_SCHEMA: Final[str] = "metamorphic_ad_verification.v1"
"""JSON schema identifier for serialised registry payloads."""

METAMORPHIC_AD_CLAIM_BOUNDARY: Final[str] = (
    "metamorphic AD verification catalogue only; executable_local laws are pure "
    "or local residual checks, permanent_boundary and refuse_invent_green rows "
    "never promote hardware formal-proof or silent-wrong recovery claims"
)
"""Shared claim boundary for catalogue rows and check results."""


@dataclass(frozen=True, slots=True)
class MetamorphicLawRecord:
    """One metamorphic / formal-boundary catalogue entry.

    Attributes
    ----------
    law_id
        Stable taxonomy key.
    kind
        Law family.
    expected_outcome
        Governance outcome for the law.
    relation
        Short statement of the metamorphic relation or boundary.
    evidence_modules
        In-tree modules that implement related suites (compose, not fork).
    reason
        Required for non-executable rows; empty only for ``executable_local``.
    default_tolerance
        Default residual band for pure residual checks (when applicable).
    claim_boundary
        Non-promotional claim boundary.

    """

    law_id: str
    kind: LawKind
    expected_outcome: LawOutcome
    relation: str
    evidence_modules: tuple[str, ...]
    reason: str = ""
    default_tolerance: float = 1e-8
    claim_boundary: str = METAMORPHIC_AD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate catalogue-entry invariants."""
        if not self.law_id or not self.law_id.strip():
            raise ValueError("law_id must be non-empty")
        if self.kind not in {
            "metamorphic_identity",
            "fd_agreement_band",
            "anti_silent_wrong",
            "formal_boundary",
        }:
            raise ValueError(f"unknown law kind: {self.kind!r}")
        if self.expected_outcome not in {
            "executable_local",
            "evidence_gated",
            "permanent_boundary",
            "refuse_invent_green",
        }:
            raise ValueError(f"unknown expected_outcome: {self.expected_outcome!r}")
        if not self.relation or not self.relation.strip():
            raise ValueError("relation must be non-empty")
        if any(not item or not item.strip() for item in self.evidence_modules):
            raise ValueError("evidence_modules must be non-empty strings")
        if self.default_tolerance <= 0.0:
            raise ValueError("default_tolerance must be positive")
        if self.expected_outcome == "executable_local":
            if self.reason:
                raise ValueError("executable_local laws must not carry a non-empty reason")
        elif not self.reason or not self.reason.strip():
            raise ValueError(
                f"non-executable laws require a non-empty reason (law_id={self.law_id!r})"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this law record."""
        return {
            "law_id": self.law_id,
            "kind": self.kind,
            "expected_outcome": self.expected_outcome,
            "relation": self.relation,
            "evidence_modules": list(self.evidence_modules),
            "reason": self.reason,
            "default_tolerance": self.default_tolerance,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MetamorphicCheckResult:
    """Result of probing a law or evaluating a pure residual band.

    Attributes
    ----------
    law_id
        Law identifier under test.
    passed
        Whether the pure check passed (False for refuse/boundary probes).
    residual
        Absolute residual for pure checks; ``None`` for catalogue-only probes.
    tolerance
        Band applied when residual is present.
    message
        Operator-facing decision message.
    refused
        True when the path is fail-closed refuse / invent-green blocked.

    """

    law_id: str
    passed: bool
    residual: float | None
    tolerance: float | None
    message: str
    refused: bool = False
    claim_boundary: str = METAMORPHIC_AD_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate check-result invariants."""
        if not self.law_id or not self.law_id.strip():
            raise ValueError("law_id must be non-empty")
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")
        if self.passed and self.refused:
            raise ValueError("a refused check cannot be marked passed")
        if self.residual is not None and self.residual < 0.0:
            raise ValueError("residual must be non-negative when provided")
        if self.tolerance is not None and self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this check result."""
        return {
            "law_id": self.law_id,
            "passed": self.passed,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "message": self.message,
            "refused": self.refused,
            "claim_boundary": self.claim_boundary,
        }


def _law(
    law_id: str,
    kind: LawKind,
    expected_outcome: LawOutcome,
    relation: str,
    *,
    evidence_modules: Sequence[str],
    reason: str = "",
    default_tolerance: float = 1e-8,
) -> MetamorphicLawRecord:
    """Build one validated catalogue row."""
    return MetamorphicLawRecord(
        law_id=law_id,
        kind=kind,
        expected_outcome=expected_outcome,
        relation=relation,
        evidence_modules=tuple(evidence_modules),
        reason=reason,
        default_tolerance=default_tolerance,
    )


_CANONICAL_LAWS: Final[tuple[MetamorphicLawRecord, ...]] = (
    _law(
        "law:metamorphic.linearity",
        "metamorphic_identity",
        "executable_local",
        "For linear maps, residual |f(a)+f(b)-f(a+b)| must stay within tolerance.",
        evidence_modules=(
            "scpn_quantum_control.differentiable_transform_algebra",
            "scpn_quantum_control.metamorphic_ad_verification",
        ),
        default_tolerance=1e-12,
    ),
    _law(
        "law:metamorphic.chain_rule_scalar",
        "metamorphic_identity",
        "executable_local",
        "Scalar chain residual |g'(f(x))*f'(x) - (g∘f)'(x)| within tolerance.",
        evidence_modules=(
            "scpn_quantum_control.differentiable_transform_algebra",
            "scpn_quantum_control.metamorphic_ad_verification",
        ),
        default_tolerance=1e-10,
    ),
    _law(
        "law:metamorphic.grad_vmap_composition",
        "metamorphic_identity",
        "evidence_gated",
        "grad(vmap(f)) composition audited by transform-algebra suite.",
        evidence_modules=("scpn_quantum_control.differentiable_transform_algebra",),
        reason=(
            "Full transform-algebra audit is evidence-gated; catalogue entry does "
            "not invent green without running the suite"
        ),
    ),
    _law(
        "law:metamorphic.jvp_vjp_duality",
        "metamorphic_identity",
        "evidence_gated",
        "JVP/VJP duality residual band under transform-algebra audit.",
        evidence_modules=("scpn_quantum_control.differentiable_transform_algebra",),
        reason=(
            "Duality is enforced by the transform-algebra audit; this entry maps "
            "the claim to evidence without re-implementing the suite"
        ),
    ),
    _law(
        "law:fd.agreement_band.parameter_shift",
        "fd_agreement_band",
        "evidence_gated",
        "Parameter-shift vs finite-difference residual bands for local kernels.",
        evidence_modules=(
            "scpn_quantum_control.differentiable_finite_difference",
            "scpn_quantum_control.differentiable_transform_algebra",
        ),
        reason=(
            "FD agreement bands are method/dtype specific; register documents the "
            "gate without inventing universal hardware FD claims"
        ),
        default_tolerance=1e-5,
    ),
    _law(
        "law:anti_silent.complex_without_wirtinger",
        "anti_silent_wrong",
        "permanent_boundary",
        "Complex reverse-mode without Wirtinger contract must refuse silently wrong real grads.",
        evidence_modules=(
            "scpn_quantum_control.unsuitable_scenario_registry",
            "scpn_quantum_control.governed_route_matrix",
            "scpn_quantum_control.differentiable_transform_algebra",
        ),
        reason=(
            "Silent real-gradient substitution on complex objectives is refused; "
            "pairs the unsuitable:complex.objective_without_wirtinger scenario"
        ),
    ),
    _law(
        "law:anti_silent.di_jl_compiled_tape",
        "anti_silent_wrong",
        "permanent_boundary",
        "DifferentiationInterface.jl compiled-tape silent-wrong class is a refuse fixture.",
        evidence_modules=(
            "scpn_quantum_control.unsuitable_scenario_registry",
            "scpn_quantum_control.governed_route_matrix",
        ),
        reason=(
            "Competitor silent-wrong reverse-mode class is permanent_boundary; "
            "SCPN must not invent silent recovery"
        ),
    ),
    _law(
        "law:formal.hardware_interactive_proof",
        "formal_boundary",
        "refuse_invent_green",
        "Interactive theorem proof of live hardware gradients is out of scope.",
        evidence_modules=(
            "scpn_quantum_control.metamorphic_ad_verification",
            "docs/internal/differentiable_programming/p3_strategic/"
            "bl46_formal_metamorphic_ad_verification.md",
        ),
        reason=(
            "Full interactive theorem proving of hardware is out of scope; refuse "
            "invent-green formal-verification marketing"
        ),
    ),
)


def _catalogue_map() -> dict[str, MetamorphicLawRecord]:
    """Return the law_id → record map for the canonical catalogue."""
    mapping = {row.law_id: row for row in _CANONICAL_LAWS}
    if len(mapping) != len(_CANONICAL_LAWS):
        raise RuntimeError("duplicate law_id in metamorphic AD verification catalogue")
    return mapping


_LAW_BY_ID: Final[Mapping[str, MetamorphicLawRecord]] = _catalogue_map()


def list_metamorphic_law_ids() -> tuple[str, ...]:
    """Return all canonical law identifiers in stable catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered law identifiers.

    """
    return tuple(row.law_id for row in _CANONICAL_LAWS)


def get_metamorphic_law(law_id: str) -> MetamorphicLawRecord:
    """Return one catalogue row or raise for unknown identifiers.

    Parameters
    ----------
    law_id
        Law taxonomy key.

    Returns
    -------
    MetamorphicLawRecord
        Matching catalogue row.

    Raises
    ------
    ValueError
        If ``law_id`` is blank or unknown.

    """
    if not law_id or not str(law_id).strip():
        raise ValueError("law_id must be a non-empty string")
    key = str(law_id).strip()
    try:
        return _LAW_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown metamorphic law_id {key!r}; refuse invent-green formal claims "
            f"(known_count={len(_LAW_BY_ID)})"
        ) from exc


def iter_metamorphic_laws(
    *,
    kind: LawKind | None = None,
    expected_outcome: LawOutcome | None = None,
) -> tuple[MetamorphicLawRecord, ...]:
    """Return filtered catalogue rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    expected_outcome
        Optional outcome filter.

    Returns
    -------
    tuple[MetamorphicLawRecord, ...]
        Matching rows.

    """
    rows: Iterable[MetamorphicLawRecord] = _CANONICAL_LAWS
    if kind is not None:
        rows = (row for row in rows if row.kind == kind)
    if expected_outcome is not None:
        rows = (row for row in rows if row.expected_outcome == expected_outcome)
    return tuple(rows)


def build_metamorphic_ad_registry() -> dict[str, object]:
    """Build the full serialisable metamorphic AD verification registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every catalogue cell (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_LAWS]
    counts: dict[str, int] = {
        "executable_local": 0,
        "evidence_gated": 0,
        "permanent_boundary": 0,
        "refuse_invent_green": 0,
    }
    for row in _CANONICAL_LAWS:
        counts[row.expected_outcome] += 1
    return {
        "schema": METAMORPHIC_AD_VERIFICATION_SCHEMA,
        "claim_boundary": METAMORPHIC_AD_CLAIM_BOUNDARY,
        "law_count": len(rows),
        "outcome_counts": counts,
        "blank_entry_count": 0,
        "laws": rows,
    }


def probe_metamorphic_law(
    law_id: str,
    *,
    unknown_policy: Literal["raise", "refuse"] = "raise",
) -> MetamorphicCheckResult:
    """Probe one catalogue law without inventing green formal-proof claims.

    Parameters
    ----------
    law_id
        Law taxonomy key.
    unknown_policy
        ``raise`` (default) rejects unknown IDs; ``refuse`` returns a refuse
        result for operator inspection.

    Returns
    -------
    MetamorphicCheckResult
        Deterministic probe metadata (not a hardware formal proof).

    Raises
    ------
    ValueError
        If ``law_id`` is blank, or unknown under ``unknown_policy='raise'``.

    """
    if not law_id or not str(law_id).strip():
        raise ValueError("law_id must be a non-empty string")
    key = str(law_id).strip()
    record = _LAW_BY_ID.get(key)
    if record is None:
        if unknown_policy == "raise":
            raise ValueError(
                f"unknown metamorphic law_id {key!r}; refuse invent-green formal claims"
            )
        if unknown_policy != "refuse":
            raise ValueError(
                f"unknown_policy must be 'raise' or 'refuse' (got {unknown_policy!r})"
            )
        return MetamorphicCheckResult(
            law_id=key,
            passed=False,
            residual=None,
            tolerance=None,
            message=(f"unknown law_id {key!r}; refuse invent-green formal-verification claims"),
            refused=True,
        )

    if record.expected_outcome in {"permanent_boundary", "refuse_invent_green"}:
        return MetamorphicCheckResult(
            law_id=key,
            passed=False,
            residual=None,
            tolerance=None,
            message=f"{record.expected_outcome}: {record.reason}",
            refused=True,
        )

    if record.expected_outcome == "evidence_gated":
        return MetamorphicCheckResult(
            law_id=key,
            passed=False,
            residual=None,
            tolerance=record.default_tolerance,
            message=(
                "evidence_gated: catalogue maps claim to suite evidence; "
                f"not invent-green without audit — {record.reason}"
            ),
            refused=False,
        )

    # executable_local: catalogue probe documents readiness; pure residual APIs
    # evaluate concrete numbers separately.
    return MetamorphicCheckResult(
        law_id=key,
        passed=True,
        residual=None,
        tolerance=record.default_tolerance,
        message=(
            "executable_local law is registered for pure residual checks "
            f"(default_tolerance={record.default_tolerance})"
        ),
        refused=False,
    )


def evaluate_linearity_residual(
    f_a: float,
    f_b: float,
    f_ab: float,
    *,
    law_id: str = "law:metamorphic.linearity",
    tolerance: float | None = None,
) -> MetamorphicCheckResult:
    """Evaluate the pure additive-linearity metamorphic residual.

    Parameters
    ----------
    f_a, f_b, f_ab
        Function values at ``a``, ``b``, and ``a+b`` for a purported linear map.
    law_id
        Must resolve to the linearity law (or raise).
    tolerance
        Optional residual band; defaults to the catalogue law tolerance.

    Returns
    -------
    MetamorphicCheckResult
        Pass/fail under the residual band.

    Raises
    ------
    ValueError
        If ``law_id`` is not the linearity executable law, or values are non-finite.

    """
    record = get_metamorphic_law(law_id)
    if record.law_id != "law:metamorphic.linearity":
        raise ValueError(
            f"evaluate_linearity_residual requires law:metamorphic.linearity (got {law_id!r})"
        )
    for name, value in (("f_a", f_a), ("f_b", f_b), ("f_ab", f_ab)):
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
            raise ValueError(f"{name} must be a finite float")
    band = record.default_tolerance if tolerance is None else float(tolerance)
    if band <= 0.0:
        raise ValueError("tolerance must be positive")
    residual = abs((f_a + f_b) - f_ab)
    passed = residual <= band
    return MetamorphicCheckResult(
        law_id=record.law_id,
        passed=passed,
        residual=residual,
        tolerance=band,
        message=(f"linearity residual={residual} tolerance={band} {'pass' if passed else 'fail'}"),
        refused=False,
    )


def evaluate_chain_rule_residual(
    outer_at_inner: float,
    inner_derivative: float,
    composite_derivative: float,
    *,
    law_id: str = "law:metamorphic.chain_rule_scalar",
    tolerance: float | None = None,
) -> MetamorphicCheckResult:
    """Evaluate the pure scalar chain-rule residual.

    Parameters
    ----------
    outer_at_inner
        ``g'(f(x))``.
    inner_derivative
        ``f'(x)``.
    composite_derivative
        Claimed ``(g∘f)'(x)``.
    law_id
        Must resolve to the chain-rule executable law.
    tolerance
        Optional residual band.

    Returns
    -------
    MetamorphicCheckResult
        Pass/fail under the residual band.

    Raises
    ------
    ValueError
        If ``law_id`` is wrong or values are non-finite / non-positive tolerance.

    """
    record = get_metamorphic_law(law_id)
    if record.law_id != "law:metamorphic.chain_rule_scalar":
        raise ValueError(
            "evaluate_chain_rule_residual requires law:metamorphic.chain_rule_scalar "
            f"(got {law_id!r})"
        )
    for name, value in (
        ("outer_at_inner", outer_at_inner),
        ("inner_derivative", inner_derivative),
        ("composite_derivative", composite_derivative),
    ):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} must be a finite float")
    band = record.default_tolerance if tolerance is None else float(tolerance)
    if band <= 0.0:
        raise ValueError("tolerance must be positive")
    expected = outer_at_inner * inner_derivative
    residual = abs(expected - composite_derivative)
    passed = residual <= band
    return MetamorphicCheckResult(
        law_id=record.law_id,
        passed=passed,
        residual=residual,
        tolerance=band,
        message=(
            f"chain_rule residual={residual} tolerance={band} {'pass' if passed else 'fail'}"
        ),
        refused=False,
    )


def assert_metamorphic_registry_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry payload contains zero blank entries.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_metamorphic_ad_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If blank entries or count drift are detected.

    """
    registry = dict(payload) if payload is not None else build_metamorphic_ad_registry()
    laws = registry.get("laws")
    if not isinstance(laws, list) or not laws:
        raise ValueError("metamorphic AD registry must contain a non-empty laws list")
    blank = 0
    for index, row in enumerate(laws):
        if not isinstance(row, Mapping):
            raise ValueError(f"law row {index} must be a mapping")
        law_id = row.get("law_id")
        outcome = row.get("expected_outcome")
        if not law_id:
            blank += 1
            continue
        if outcome not in {
            "executable_local",
            "evidence_gated",
            "permanent_boundary",
            "refuse_invent_green",
        }:
            blank += 1
            continue
        if outcome != "executable_local" and not row.get("reason"):
            raise ValueError(f"law {law_id!r} is non-executable without reason")
    if blank:
        raise ValueError(
            f"metamorphic AD registry has {blank} blank or invalid entries; refuse green"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    law_count = registry.get("law_count", -1)
    if not isinstance(law_count, int) or law_count != len(laws):
        raise ValueError("law_count does not match laws list length")
    return registry


__all__ = [
    "METAMORPHIC_AD_CLAIM_BOUNDARY",
    "METAMORPHIC_AD_VERIFICATION_SCHEMA",
    "LawKind",
    "LawOutcome",
    "MetamorphicCheckResult",
    "MetamorphicLawRecord",
    "assert_metamorphic_registry_integrity",
    "build_metamorphic_ad_registry",
    "evaluate_chain_rule_residual",
    "evaluate_linearity_residual",
    "get_metamorphic_law",
    "iter_metamorphic_laws",
    "list_metamorphic_law_ids",
    "probe_metamorphic_law",
]
