# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust Program AD fuzz assurance product
"""Fail-closed **Rust Program AD fuzz assurance** product surface.

Provides fuzz assurance for bit-exact Program AD replay: a versioned
fuzz target catalogue over ambient ``scpn_quantum_engine/fuzz`` bins, time-boxed
CI-optional policy, dry-run / probe helpers that refuse invent-green continuous
multi-hour cargo-fuzz coverage claims.

Does **not** execute cargo-fuzz or invent continuous corpus green status.
Multi-day corpus retention, automated crash-to-regression conversion, and
parity-certificate fuzz-case ingestion remain unimplemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

TargetKind = Literal[
    "program_ad_ir",
    "studio_kuramoto_input",
    "ml_dsa_ntt",
    "knm_validators",
    "policy",
]
"""Kinds of catalogue entries (fuzz bins + policy row)."""

TargetPosture = Literal[
    "time_boxed_local",
    "ci_optional",
    "continuous_forbidden_default",
]
"""Fuzz execution posture honesty labels."""

ProbeOutcome = Literal["allowed_dry_run", "refused"]
"""Structured dry-run / probe outcomes."""

PROGRAM_AD_FUZZ_ASSURANCE_SCHEMA: Final[str] = "program_ad_fuzz_assurance.v2"
"""JSON schema identifier for serialised product payloads."""

DEFAULT_TIME_BOX_SECONDS: Final[int] = 300
"""Default time-box for optional local/CI fuzz runs (5 minutes)."""

MAX_TIME_BOX_SECONDS: Final[int] = 3600
"""Hard upper bound for product-declared time boxes (1 hour); not continuous."""

PROGRAM_AD_FUZZ_CLAIM_BOUNDARY: Final[str] = (
    "Rust Program AD fuzz assurance product only; catalogues ambient "
    "scpn_quantum_engine/fuzz targets and time-boxed CI-optional policy; "
    "does not execute cargo-fuzz or invent-green continuous multi-hour "
    "coverage; multi-day corpus retention, automated crash-to-regression "
    "conversion, and parity-certificate fuzz-case ingestion remain unimplemented"
)
"""Shared claim boundary for targets, policy, and probe decisions."""


@dataclass(frozen=True, slots=True)
class FuzzTarget:
    """One ambient cargo-fuzz target in the product catalogue.

    Attributes
    ----------
    target_id
        Stable catalogue identifier (matches cargo-fuzz bin name).
    title
        Human-readable title.
    summary
        Short description of what the harness exercises.
    kind
        Target kind classification.
    rust_path
        Path to the fuzz target source relative to repo root.
    package
        Cargo package name owning the fuzz bin.
    posture
        Declared execution posture.
    parity_certificate_pointer
        Optional parity-certificate feed pointer (residual when empty feed).
    api_stability_class
        Stability honesty class.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    target_id: str
    title: str
    summary: str
    kind: TargetKind
    rust_path: str
    package: str
    posture: TargetPosture = "time_boxed_local"
    parity_certificate_pointer: str = "polyglot_parity_certificate.fuzz_case_feed_residual"
    api_stability_class: str = "experimental_workbench"
    as_of: str = "2026-07-24"
    claim_boundary: str = PROGRAM_AD_FUZZ_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate target invariants."""
        if not self.target_id or not self.target_id.strip():
            raise ValueError("target_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.kind not in {
            "program_ad_ir",
            "studio_kuramoto_input",
            "ml_dsa_ntt",
            "knm_validators",
            "policy",
        }:
            raise ValueError(f"unknown target kind: {self.kind!r}")
        if not self.rust_path or not self.rust_path.strip():
            raise ValueError("rust_path must be non-empty")
        if not self.package or not self.package.strip():
            raise ValueError("package must be non-empty")
        if self.posture not in {
            "time_boxed_local",
            "ci_optional",
            "continuous_forbidden_default",
        }:
            raise ValueError(f"unknown posture: {self.posture!r}")
        if not self.api_stability_class or not self.api_stability_class.strip():
            raise ValueError("api_stability_class must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this target."""
        return {
            "target_id": self.target_id,
            "title": self.title,
            "summary": self.summary,
            "kind": self.kind,
            "rust_path": self.rust_path,
            "package": self.package,
            "posture": self.posture,
            "parity_certificate_pointer": self.parity_certificate_pointer,
            "api_stability_class": self.api_stability_class,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class FuzzPolicy:
    """Time-boxed / CI-optional fuzz policy.

    Attributes
    ----------
    policy_id
        Stable policy identifier.
    default_time_box_seconds
        Default time box for optional runs.
    max_time_box_seconds
        Hard upper bound declared by the product (not continuous).
    continuous_fuzz_default
        Whether continuous multi-hour fuzz is the default (must be False).
    ci_optional
        Whether CI may opt into a time-boxed job.
    invent_green_forbidden
        Whether invent-green continuous coverage is forbidden (must be True).
    claim_boundary
        Non-promotional claim boundary.

    """

    policy_id: str
    default_time_box_seconds: int
    max_time_box_seconds: int
    continuous_fuzz_default: bool
    ci_optional: bool
    invent_green_forbidden: bool
    claim_boundary: str = PROGRAM_AD_FUZZ_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate policy invariants."""
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if self.default_time_box_seconds <= 0:
            raise ValueError("default_time_box_seconds must be positive")
        if self.max_time_box_seconds <= 0:
            raise ValueError("max_time_box_seconds must be positive")
        if self.default_time_box_seconds > self.max_time_box_seconds:
            raise ValueError("default_time_box_seconds cannot exceed max_time_box_seconds")
        if self.continuous_fuzz_default:
            raise ValueError("continuous_fuzz_default must be False (no invent-green continuous)")
        if not self.invent_green_forbidden:
            raise ValueError("invent_green_forbidden must be True")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this policy."""
        return {
            "policy_id": self.policy_id,
            "default_time_box_seconds": self.default_time_box_seconds,
            "max_time_box_seconds": self.max_time_box_seconds,
            "continuous_fuzz_default": self.continuous_fuzz_default,
            "ci_optional": self.ci_optional,
            "invent_green_forbidden": self.invent_green_forbidden,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class FuzzProbeDecision:
    """Fail-closed dry-run decision for a fuzz target probe.

    Attributes
    ----------
    target_id
        Target probed.
    outcome
        Allowed dry-run or refused.
    allowed
        Whether a time-boxed dry-run plan may be acknowledged.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    time_box_seconds
        Time box acknowledged (0 when refused).

    """

    target_id: str
    outcome: ProbeOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    time_box_seconds: int
    claim_boundary: str = PROGRAM_AD_FUZZ_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate probe decision invariants."""
        if not self.target_id or not self.target_id.strip():
            raise ValueError("target_id must be non-empty")
        if self.outcome not in {"allowed_dry_run", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed_dry_run":
            raise ValueError("allowed decisions must use outcome=allowed_dry_run")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.time_box_seconds < 0:
            raise ValueError("time_box_seconds must be non-negative")
        if self.allowed and self.time_box_seconds <= 0:
            raise ValueError("allowed decisions require positive time_box_seconds")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "target_id": self.target_id,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "time_box_seconds": self.time_box_seconds,
            "claim_boundary": self.claim_boundary,
        }


def _target(
    target_id: str,
    *,
    title: str,
    summary: str,
    kind: TargetKind,
    rust_path: str,
    posture: TargetPosture = "time_boxed_local",
) -> FuzzTarget:
    """Build one catalogue target."""
    return FuzzTarget(
        target_id=target_id,
        title=title,
        summary=summary,
        kind=kind,
        rust_path=rust_path,
        package="scpn-quantum-engine-fuzz",
        posture=posture,
    )


_CANONICAL_TARGETS: Final[tuple[FuzzTarget, ...]] = (
    _target(
        "program_ad_ir",
        title="Program AD IR parse + bounded replay",
        summary=(
            "Coverage-guided fuzz of Program AD effect-IR parse, forward "
            "replay, and value+gradient replay (capped IR byte size)."
        ),
        kind="program_ad_ir",
        rust_path="scpn_quantum_engine/fuzz/fuzz_targets/program_ad_ir.rs",
        posture="ci_optional",
    ),
    _target(
        "studio_kuramoto_input",
        title="Studio Kuramoto input validators",
        summary=(
            "Fuzz harness for studio Kuramoto input validation paths used by WASM/studio kernels."
        ),
        kind="studio_kuramoto_input",
        rust_path="scpn_quantum_engine/fuzz/fuzz_targets/studio_kuramoto_input.rs",
        posture="time_boxed_local",
    ),
    _target(
        "ml_dsa_ntt",
        title="ML-DSA NTT/INTT domain bijection",
        summary=(
            "Fuzz harness asserting NTT/INTT round-trip bijection on [0, q) "
            "for arbitrary coefficient vectors."
        ),
        kind="ml_dsa_ntt",
        rust_path="scpn_quantum_engine/fuzz/fuzz_targets/ml_dsa_ntt.rs",
        posture="time_boxed_local",
    ),
    _target(
        "knm_validators",
        title="K_nm shared input validators",
        summary=(
            "Fuzz harness for shared Rust input validators (check_finite, "
            "check_n, domain range) and bounded build_knm_inner replay."
        ),
        kind="knm_validators",
        rust_path="scpn_quantum_engine/fuzz/fuzz_targets/knm_validators.rs",
        posture="ci_optional",
    ),
)

_DEFAULT_POLICY: Final[FuzzPolicy] = FuzzPolicy(
    policy_id="time_boxed_ci_optional_v1",
    default_time_box_seconds=DEFAULT_TIME_BOX_SECONDS,
    max_time_box_seconds=MAX_TIME_BOX_SECONDS,
    continuous_fuzz_default=False,
    ci_optional=True,
    invent_green_forbidden=True,
)


def _catalogue_map() -> dict[str, FuzzTarget]:
    """Return target_id → target map; refuse blanks/duplicates."""
    mapping: dict[str, FuzzTarget] = {}
    for row in _CANONICAL_TARGETS:
        key = row.target_id.strip()
        if not key:
            raise RuntimeError("fuzz assurance catalogue contains blank target_id")
        if key in mapping:
            raise RuntimeError(f"duplicate target_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("fuzz assurance catalogue must be non-empty")
    return mapping


_TARGET_BY_ID: Final[Mapping[str, FuzzTarget]] = _catalogue_map()


def list_fuzz_target_ids() -> tuple[str, ...]:
    """Return all fuzz target identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered target identifiers.

    """
    return tuple(row.target_id for row in _CANONICAL_TARGETS)


def get_fuzz_target(target_id: str) -> FuzzTarget:
    """Return one target or raise for blank/unknown identifiers.

    Parameters
    ----------
    target_id
        Catalogue target key.

    Returns
    -------
    FuzzTarget
        Matching target.

    Raises
    ------
    ValueError
        If ``target_id`` is blank or unknown (fail closed).

    """
    if not target_id or not str(target_id).strip():
        raise ValueError("target_id must be a non-empty string")
    key = str(target_id).strip()
    try:
        return _TARGET_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown target_id {key!r}; refuse invent-green fuzz assurance "
            f"claim (known_count={len(_TARGET_BY_ID)})"
        ) from exc


def iter_fuzz_targets(
    *,
    posture: TargetPosture | None = None,
    kind: TargetKind | None = None,
) -> tuple[FuzzTarget, ...]:
    """Return filtered targets in stable order.

    Parameters
    ----------
    posture
        Optional posture filter.
    kind
        Optional kind filter.

    Returns
    -------
    tuple[FuzzTarget, ...]
        Matching targets.

    """
    rows: Sequence[FuzzTarget] = _CANONICAL_TARGETS
    if posture is not None:
        rows = tuple(row for row in rows if row.posture == posture)
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def fuzz_assurance_policy() -> FuzzPolicy:
    """Return the default time-boxed / CI-optional fuzz policy.

    Returns
    -------
    FuzzPolicy
        Product policy (continuous default false; invent-green forbidden).

    """
    return _DEFAULT_POLICY


def validate_time_box_seconds(time_box_seconds: int) -> int:
    """Validate a requested time box against product policy.

    Parameters
    ----------
    time_box_seconds
        Requested duration in seconds.

    Returns
    -------
    int
        Accepted time box.

    Raises
    ------
    ValueError
        If non-positive, non-int, or exceeds max time box (continuous refuse).

    """
    if not isinstance(time_box_seconds, int) or isinstance(time_box_seconds, bool):
        raise ValueError("time_box_seconds must be an int")
    if time_box_seconds <= 0:
        raise ValueError("time_box_seconds must be positive")
    policy = fuzz_assurance_policy()
    if time_box_seconds > policy.max_time_box_seconds:
        raise ValueError(
            f"time_box_seconds {time_box_seconds} exceeds max "
            f"{policy.max_time_box_seconds}; refuse invent-green continuous fuzz"
        )
    return time_box_seconds


def dry_run_fuzz_target(
    target_id: str,
    *,
    time_box_seconds: int | None = None,
    request_continuous: bool = False,
    request_invent_green_coverage: bool = False,
) -> FuzzProbeDecision:
    """Acknowledge a time-boxed fuzz dry-run plan without executing cargo-fuzz.

    Parameters
    ----------
    target_id
        Catalogue target key.
    time_box_seconds
        Optional time box (defaults to policy default).
    request_continuous
        When true, refuse (no continuous multi-hour invent-green).
    request_invent_green_coverage
        When true, refuse invent-green continuous coverage claims.

    Returns
    -------
    FuzzProbeDecision
        Allowed dry-run or refused decision.

    Raises
    ------
    ValueError
        If ``target_id`` is blank/unknown or time box is invalid (when not
        refused via continuous flags first).

    """
    target = get_fuzz_target(target_id)
    policy = fuzz_assurance_policy()
    blockers: list[str] = []

    if request_continuous:
        blockers.append(
            "continuous multi-hour cargo-fuzz request refused "
            "(product continuous_fuzz_default=False; invent-green forbidden)"
        )
    if request_invent_green_coverage:
        blockers.append(
            "invent-green continuous fuzz coverage claim refused "
            f"(policy {policy.policy_id!r} invent_green_forbidden=True)"
        )

    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return FuzzProbeDecision(
            target_id=target.target_id,
            outcome="refused",
            allowed=False,
            reason="fuzz assurance product refuse: " + "; ".join(unique),
            blockers=unique,
            time_box_seconds=0,
        )

    seconds = (
        policy.default_time_box_seconds
        if time_box_seconds is None
        else validate_time_box_seconds(time_box_seconds)
    )
    return FuzzProbeDecision(
        target_id=target.target_id,
        outcome="allowed_dry_run",
        allowed=True,
        reason=(
            f"time-boxed dry-run plan for {target.target_id!r} allowed "
            f"({seconds}s, posture={target.posture}, package={target.package}); "
            "cargo-fuzz was not executed; no continuous invent-green coverage claimed"
        ),
        blockers=(),
        time_box_seconds=seconds,
    )


def corpus_governance_policy() -> dict[str, object]:
    """Return the corpus-governance implementation boundary.

    Returns
    -------
    dict[str, object]
        Policy describing the unimplemented corpus-retention automation.

    """
    return {
        "policy_id": "corpus_governance_boundary_v1",
        "ambient_corpus_path": "scpn_quantum_engine/fuzz/corpus",
        "ambient_artifacts_path": "scpn_quantum_engine/fuzz/artifacts",
        "retention_ops_implemented": False,
        "open_capability": "multi_day_corpus_retention",
        "claim_boundary": (
            "Corpus paths are ambient inventory pointers only; multi-day "
            "retention automation remains unimplemented; product does not invent "
            "retention green status"
        ),
    }


def crash_pipeline_policy() -> dict[str, object]:
    """Return the crash-to-regression implementation boundary.

    Returns
    -------
    dict[str, object]
        Policy describing the unimplemented crash automation.

    """
    return {
        "policy_id": "crash_regression_pipeline_boundary_v1",
        "automated_pipeline_implemented": False,
        "open_capability": "automated_crash_to_regression_conversion",
        "claim_boundary": (
            "Crash-to-regression automation remains unimplemented; product does "
            "not invent-green a live crash triage pipeline"
        ),
    }


def map_fuzz_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of fuzz assurance product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.program_ad_fuzz_assurance",
            "role": "program_ad_fuzz_assurance_product",
            "api_stability_class": "experimental_workbench",
            "target_ids": list(list_fuzz_target_ids()),
            "claim_boundary": PROGRAM_AD_FUZZ_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_engine.fuzz",
            "role": "ambient_cargo_fuzz_package",
            "api_stability_class": "experimental_workbench",
            "target_ids": list(list_fuzz_target_ids()),
            "claim_boundary": PROGRAM_AD_FUZZ_CLAIM_BOUNDARY,
        },
    )


def build_fuzz_assurance_registry() -> dict[str, object]:
    """Build the full serialisable fuzz assurance product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with targets and policy (no blanks).

    """
    targets = [row.to_dict() for row in _CANONICAL_TARGETS]
    policy = fuzz_assurance_policy().to_dict()
    return {
        "schema": PROGRAM_AD_FUZZ_ASSURANCE_SCHEMA,
        "claim_boundary": PROGRAM_AD_FUZZ_CLAIM_BOUNDARY,
        "target_count": len(targets),
        "blank_entry_count": 0,
        "default_target_id": "program_ad_ir",
        "policy": policy,
        "corpus_governance": corpus_governance_policy(),
        "crash_pipeline": crash_pipeline_policy(),
        "public_surfaces": list(map_fuzz_public_surfaces()),
        "targets": targets,
        "policy_note": (
            "Fuzz assurance product catalogue only; ambient cargo-fuzz bins "
            "are not executed by this module; continuous multi-hour invent-green "
            "coverage is forbidden; corpus retention, crash conversion, and "
            "parity-certificate ingestion remain unimplemented."
        ),
    }


def assert_fuzz_assurance_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers targets without blanks or invent-green policy.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_fuzz_assurance_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green policy appear.

    """
    registry = dict(payload) if payload is not None else build_fuzz_assurance_registry()
    targets = registry.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("fuzz assurance registry must contain a non-empty targets list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(targets):
        if not isinstance(row, Mapping):
            raise ValueError(f"target row {index} must be a mapping")
        target_id = row.get("target_id")
        posture = row.get("posture")
        rust_path = row.get("rust_path")
        if not target_id or not str(target_id).strip():
            blank += 1
            continue
        tid = str(target_id).strip()
        if tid in seen:
            raise ValueError(f"duplicate target_id in registry: {tid!r}")
        seen.add(tid)
        if tid == "program_ad_ir":
            default_found = True
        if posture not in {
            "time_boxed_local",
            "ci_optional",
            "continuous_forbidden_default",
        }:
            blank += 1
            continue
        if not rust_path or not str(rust_path).strip():
            raise ValueError(f"target {tid!r} must have rust_path")
    if blank:
        raise ValueError(f"fuzz assurance registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("fuzz assurance registry missing program_ad_ir")
    expected = set(list_fuzz_target_ids())
    if seen != expected:
        raise ValueError(
            f"registry target set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    target_count = registry.get("target_count", -1)
    if not isinstance(target_count, int) or target_count != len(targets):
        raise ValueError("target_count does not match targets list length")
    policy = registry.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be a mapping")
    if policy.get("continuous_fuzz_default") is not False:
        raise ValueError("policy.continuous_fuzz_default must be False")
    if policy.get("invent_green_forbidden") is not True:
        raise ValueError("policy.invent_green_forbidden must be True")
    return registry


__all__ = [
    "DEFAULT_TIME_BOX_SECONDS",
    "MAX_TIME_BOX_SECONDS",
    "PROGRAM_AD_FUZZ_ASSURANCE_SCHEMA",
    "PROGRAM_AD_FUZZ_CLAIM_BOUNDARY",
    "FuzzPolicy",
    "FuzzProbeDecision",
    "FuzzTarget",
    "ProbeOutcome",
    "TargetKind",
    "TargetPosture",
    "assert_fuzz_assurance_integrity",
    "build_fuzz_assurance_registry",
    "corpus_governance_policy",
    "crash_pipeline_policy",
    "dry_run_fuzz_target",
    "fuzz_assurance_policy",
    "get_fuzz_target",
    "iter_fuzz_targets",
    "list_fuzz_target_ids",
    "map_fuzz_public_surfaces",
    "validate_time_box_seconds",
]
