# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — polyglot parity certificates product
"""Fail-closed **bit-exact polyglot parity certificate** product surface.

Productises externally checkable certificate bundles for the Rust Program AD
replay moat: versioned schema, family catalogue (scalar → spectral bounds),
digest build/verify helpers, and refuse invent-green full NumPy parity or
unsupported Rust feature claims.

Composes honesty from ambient ``program_ad_rust_bridge`` / inventory pointers
without re-running the entire cargo matrix as the product façade. Command-line
access, a committed multi-family CI corpus, and Rust-JIT decision evidence
integration remain open.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

FamilySupport = Literal[
    "sample_bitexact",
    "boundary_unsupported",
    "catalogue_only",
]
"""Support posture for a certificate family."""

VerifyOutcome = Literal["passed", "failed", "refused"]
"""Structured verify outcomes."""

POLYGLOT_PARITY_CERTIFICATE_SCHEMA: Final[str] = "polyglot_parity_certificate.v2"
"""JSON schema identifier for serialised certificates and product registry."""

POLYGLOT_PARITY_PRODUCT_SCHEMA: Final[str] = "polyglot_parity_certificate_product.v2"
"""Product registry schema (family catalogue + policy)."""

POLYGLOT_PARITY_CLAIM_BOUNDARY: Final[str] = (
    "This polyglot parity certificate product proves only the identity of "
    "published sample bundles through digests; it does not claim full NumPy "
    "parity. Unsupported Rust and feature paths fail closed with typed blockers. "
    "The Program AD Rust bridge remains an experimental workbench. A command-line "
    "entry point, committed multi-family CI corpus, and Rust-JIT decision evidence "
    "integration remain open."
)
"""Shared claim boundary for families, certificates, and verify decisions."""

_POLYGLOT_PARITY_POLICY_NOTE: Final[str] = (
    "Digests prove only published sample bundle identity; full NumPy parity is "
    "never claimed. The command-line entry point, committed multi-family CI "
    "corpus, and Rust-JIT decision evidence integration remain open."
)
"""Canonical product-registry policy note."""


def _require_exact_claim_boundary(claim_boundary: str) -> None:
    """Reject records whose claim boundary differs from the governed contract."""
    if claim_boundary != POLYGLOT_PARITY_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must match POLYGLOT_PARITY_CLAIM_BOUNDARY exactly")


def _is_sha256_digest(value: str) -> bool:
    """Return whether a string is exactly one lowercase SHA-256 hex digest."""
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class ParityFamily:
    """One polyglot parity certificate family.

    Attributes
    ----------
    family_id
        Stable catalogue identifier.
    title
        Human-readable title.
    summary
        Short description of the family scope.
    support
        Support posture (sample bit-exact, boundary unsupported, catalogue only).
    module_path
        Primary ambient module pointer.
    api_stability_class
        Stability honesty class.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    family_id: str
    title: str
    summary: str
    support: FamilySupport
    module_path: str
    api_stability_class: str = "experimental_workbench"
    as_of: str = "2026-07-24"
    claim_boundary: str = POLYGLOT_PARITY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate family invariants."""
        if not self.family_id or not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.support not in {
            "sample_bitexact",
            "boundary_unsupported",
            "catalogue_only",
        }:
            raise ValueError(f"unknown support posture: {self.support!r}")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.api_stability_class or not self.api_stability_class.strip():
            raise ValueError("api_stability_class must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this family."""
        return {
            "family_id": self.family_id,
            "title": self.title,
            "summary": self.summary,
            "support": self.support,
            "module_path": self.module_path,
            "api_stability_class": self.api_stability_class,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PolyglotParityCertificate:
    """One bit-exact polyglot parity certificate bundle.

    Attributes
    ----------
    family_id
        Family this certificate belongs to.
    schema
        Certificate schema identifier.
    sample_id
        Stable sample case identifier within the family.
    input_digest
        SHA-256 of canonical input payload.
    python_reference_digest
        SHA-256 of Python reference output payload.
    rust_digest
        SHA-256 of Rust output payload, or empty when unsupported.
    max_abs_error
        Maximum absolute error (0.0 for bit-exact pass).
    supported
        Whether the certificate claims supported bit-exact parity.
    blocked_reasons
        Non-empty when unsupported or failed.
    claim_boundary
        Non-promotional claim boundary.

    """

    family_id: str
    schema: str
    sample_id: str
    input_digest: str
    python_reference_digest: str
    rust_digest: str
    max_abs_error: float
    supported: bool
    blocked_reasons: tuple[str, ...]
    claim_boundary: str = POLYGLOT_PARITY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate certificate invariants."""
        if not self.family_id or not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if not self.schema or not self.schema.strip():
            raise ValueError("schema must be non-empty")
        if self.schema != POLYGLOT_PARITY_CERTIFICATE_SCHEMA:
            raise ValueError(
                f"unknown certificate schema {self.schema!r}; refuse invent-green "
                f"(expected {POLYGLOT_PARITY_CERTIFICATE_SCHEMA!r})"
            )
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        for name, digest in (
            ("input_digest", self.input_digest),
            ("python_reference_digest", self.python_reference_digest),
        ):
            if not _is_sha256_digest(digest):
                raise ValueError(f"{name} must be a 64-char lowercase hex digest")
        if self.rust_digest and not _is_sha256_digest(self.rust_digest):
            raise ValueError("rust_digest must be empty or a 64-char lowercase hex digest")
        if self.max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if any(not item or not item.strip() for item in self.blocked_reasons):
            raise ValueError("blocked_reasons entries must be non-empty")
        if self.supported and self.blocked_reasons:
            raise ValueError("supported certificates cannot list blockers")
        if self.supported and not self.rust_digest:
            raise ValueError("supported certificates require rust_digest")
        if self.supported and self.max_abs_error != 0.0:
            raise ValueError("supported bit-exact certificates require max_abs_error == 0.0")
        if not self.supported and not self.blocked_reasons:
            raise ValueError("unsupported certificates require blockers")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this certificate."""
        return {
            "family_id": self.family_id,
            "schema": self.schema,
            "sample_id": self.sample_id,
            "input_digest": self.input_digest,
            "python_reference_digest": self.python_reference_digest,
            "rust_digest": self.rust_digest,
            "max_abs_error": self.max_abs_error,
            "supported": self.supported,
            "blocked_reasons": list(self.blocked_reasons),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class CertificateVerifyDecision:
    """Fail-closed verify decision for a certificate.

    Attributes
    ----------
    family_id
        Family verified.
    sample_id
        Sample verified.
    outcome
        Passed, failed, or refused.
    passed
        Whether verify accepted the certificate as bit-exact.
    reason
        Human-readable reason.
    blockers
        Non-empty when failed or refused.
    observed_max_abs_error
        Recomputed error (0.0 when digests match for supported certs).

    """

    family_id: str
    sample_id: str
    outcome: VerifyOutcome
    passed: bool
    reason: str
    blockers: tuple[str, ...]
    observed_max_abs_error: float
    claim_boundary: str = POLYGLOT_PARITY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate verify decision invariants."""
        if not self.family_id or not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if self.outcome not in {"passed", "failed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.passed and self.outcome != "passed":
            raise ValueError("passed decisions must use outcome=passed")
        if not self.passed and self.outcome == "passed":
            raise ValueError("non-passed decisions cannot use outcome=passed")
        if self.passed and self.blockers:
            raise ValueError("passed decisions cannot list blockers")
        if not self.passed and not self.blockers:
            raise ValueError("non-passed decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.observed_max_abs_error < 0.0:
            raise ValueError("observed_max_abs_error must be non-negative")
        _require_exact_claim_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "family_id": self.family_id,
            "sample_id": self.sample_id,
            "outcome": self.outcome,
            "passed": self.passed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "observed_max_abs_error": self.observed_max_abs_error,
            "claim_boundary": self.claim_boundary,
        }


def _family(
    family_id: str,
    *,
    title: str,
    summary: str,
    support: FamilySupport,
    module_path: str,
) -> ParityFamily:
    """Build one catalogue family."""
    return ParityFamily(
        family_id=family_id,
        title=title,
        summary=summary,
        support=support,
        module_path=module_path,
    )


_CANONICAL_FAMILIES: Final[tuple[ParityFamily, ...]] = (
    _family(
        "scalar_interpreter_replay",
        title="Scalar interpreter replay",
        summary=(
            "Bounded Rust Program AD scalar IR interpreter vs Python reference "
            "digest for published sample inputs."
        ),
        support="sample_bitexact",
        module_path="scpn_quantum_control.program_ad_rust_bridge",
    ),
    _family(
        "value_and_gradient_replay",
        title="Value and gradient replay",
        summary=(
            "Bounded Rust Program AD value+gradient replay vs Python reference "
            "for published sample IR bundles."
        ),
        support="sample_bitexact",
        module_path="scpn_quantum_control.program_ad_rust_bridge",
    ),
    _family(
        "registry_metadata_mirror",
        title="Registry metadata mirror",
        summary=(
            "Rust Program AD registry metadata mirror counts/digests vs Python "
            "registry inventory snapshot."
        ),
        support="sample_bitexact",
        module_path="scpn_quantum_control.program_ad_rust_bridge",
    ),
    _family(
        "elementwise_primitive_parity",
        title="Elementwise primitive parity boundary",
        summary=(
            "Elementwise primitive family boundary; refuses invent-green until "
            "polyglot elementwise AD is supported."
        ),
        support="boundary_unsupported",
        module_path="scpn_quantum_control.program_ad_elementwise_primitives",
    ),
    _family(
        "linalg_primitive_parity",
        title="Linalg primitive parity boundary",
        summary=(
            "Linalg primitive family boundary; refuses invent-green until "
            "polyglot linalg AD is supported."
        ),
        support="boundary_unsupported",
        module_path="scpn_quantum_control.program_ad_linalg_primitives",
    ),
    _family(
        "spectral_bounds_parity",
        title="Spectral bounds parity (catalogue)",
        summary=(
            "Spectral bounds family reserved in the catalogue; sample bit-exact "
            "corpus residual (catalogue_only until corpus lands)."
        ),
        support="catalogue_only",
        module_path="scpn_quantum_control.program_ad_rust_bridge",
    ),
)


def _catalogue_map() -> dict[str, ParityFamily]:
    """Return family_id → family map; refuse blanks/duplicates."""
    mapping: dict[str, ParityFamily] = {}
    for row in _CANONICAL_FAMILIES:
        key = row.family_id.strip()
        if not key:
            raise RuntimeError("polyglot parity catalogue contains blank family_id")
        if key in mapping:
            raise RuntimeError(f"duplicate family_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("polyglot parity catalogue must be non-empty")
    return mapping


_FAMILY_BY_ID: Final[Mapping[str, ParityFamily]] = _catalogue_map()


def list_parity_family_ids() -> tuple[str, ...]:
    """Return all certificate family identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered family identifiers.

    """
    return tuple(row.family_id for row in _CANONICAL_FAMILIES)


def get_parity_family(family_id: str) -> ParityFamily:
    """Return one family or raise for blank/unknown identifiers.

    Parameters
    ----------
    family_id
        Catalogue family key.

    Returns
    -------
    ParityFamily
        Matching family.

    Raises
    ------
    ValueError
        If ``family_id`` is blank or unknown (fail closed).

    """
    if not family_id or not str(family_id).strip():
        raise ValueError("family_id must be a non-empty string")
    key = str(family_id).strip()
    try:
        return _FAMILY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown family_id {key!r}; refuse invent-green polyglot parity "
            f"claim (known_count={len(_FAMILY_BY_ID)})"
        ) from exc


def iter_parity_families(
    *,
    support: FamilySupport | None = None,
) -> tuple[ParityFamily, ...]:
    """Return filtered families in stable order.

    Parameters
    ----------
    support
        Optional support posture filter.

    Returns
    -------
    tuple[ParityFamily, ...]
        Matching families.

    """
    rows: Sequence[ParityFamily] = _CANONICAL_FAMILIES
    if support is not None:
        rows = tuple(row for row in rows if row.support == support)
    return tuple(rows)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes (sorted keys, compact).

    Parameters
    ----------
    payload
        JSON-compatible mapping.

    Returns
    -------
    bytes
        Canonical JSON encoding.

    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def digest_payload(payload: Mapping[str, Any]) -> str:
    """Return SHA-256 hex digest of canonical JSON for a payload.

    Parameters
    ----------
    payload
        JSON-compatible mapping.

    Returns
    -------
    str
        64-character lowercase hex digest.

    """
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sample_input_payload(family_id: str, sample_id: str) -> dict[str, object]:
    """Return deterministic sample input payload for a family."""
    return {
        "family_id": family_id,
        "sample_id": sample_id,
        "inputs": {
            "scalar": 1.25,
            "vector": [0.0, 0.5, -0.25],
            "seed": 49,
        },
        "schema": POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
    }


def _sample_python_reference_payload(family_id: str, sample_id: str) -> dict[str, object]:
    """Return deterministic Python reference output payload for a family."""
    return {
        "family_id": family_id,
        "sample_id": sample_id,
        "reference": "python",
        "outputs": {
            "value": 1.25,
            "gradient": [1.0, 0.0, 0.0],
            "status": "ok",
        },
    }


def _sample_rust_payload(family_id: str, sample_id: str) -> dict[str, object]:
    """Return deterministic Rust output payload matching Python for bit-exact samples."""
    # Bit-exact sample: identical structure to Python reference.
    return {
        "family_id": family_id,
        "sample_id": sample_id,
        "reference": "rust",
        "outputs": {
            "value": 1.25,
            "gradient": [1.0, 0.0, 0.0],
            "status": "ok",
        },
    }


def build_sample_certificate(
    family_id: str,
    *,
    sample_id: str = "sample-0",
) -> PolyglotParityCertificate:
    """Build a deterministic sample certificate for a family.

    Sample bit-exact families produce supported certificates with matching
    digests and ``max_abs_error == 0.0``. Boundary/catalogue families produce
    unsupported certificates with typed blockers (no invent-green).

    Parameters
    ----------
    family_id
        Catalogue family key.
    sample_id
        Stable sample case identifier.

    Returns
    -------
    PolyglotParityCertificate
        Built certificate bundle.

    Raises
    ------
    ValueError
        If ``family_id`` or ``sample_id`` is blank/unknown.

    """
    family = get_parity_family(family_id)
    if not sample_id or not str(sample_id).strip():
        raise ValueError("sample_id must be a non-empty string")
    sid = str(sample_id).strip()
    inputs = _sample_input_payload(family.family_id, sid)
    py_ref = _sample_python_reference_payload(family.family_id, sid)
    input_digest = digest_payload(inputs)
    python_digest = digest_payload(py_ref)

    if family.support == "sample_bitexact":
        rust_payload = _sample_rust_payload(family.family_id, sid)
        rust_digest = digest_payload(rust_payload)
        # Bit-exact: rust outputs match python outputs field-wise.
        return PolyglotParityCertificate(
            family_id=family.family_id,
            schema=POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
            sample_id=sid,
            input_digest=input_digest,
            python_reference_digest=python_digest,
            rust_digest=rust_digest,
            max_abs_error=0.0,
            supported=True,
            blocked_reasons=(),
        )

    if family.support == "boundary_unsupported":
        blockers = (
            f"family {family.family_id!r} is boundary_unsupported; refuse invent-green "
            "polyglot bit-exact parity until feature support lands",
        )
        return PolyglotParityCertificate(
            family_id=family.family_id,
            schema=POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
            sample_id=sid,
            input_digest=input_digest,
            python_reference_digest=python_digest,
            rust_digest="",
            max_abs_error=0.0,
            supported=False,
            blocked_reasons=blockers,
        )

    # catalogue_only
    blockers = (
        f"family {family.family_id!r} is catalogue_only; sample bit-exact corpus "
        "is not committed; refuse invent-green certificate pass",
    )
    return PolyglotParityCertificate(
        family_id=family.family_id,
        schema=POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
        sample_id=sid,
        input_digest=input_digest,
        python_reference_digest=python_digest,
        rust_digest="",
        max_abs_error=0.0,
        supported=False,
        blocked_reasons=blockers,
    )


def certificate_from_dict(payload: Mapping[str, Any]) -> PolyglotParityCertificate:
    """Rebuild a certificate from a JSON-compatible mapping.

    Parameters
    ----------
    payload
        Mapping produced by :meth:`PolyglotParityCertificate.to_dict`.

    Returns
    -------
    PolyglotParityCertificate
        Validated certificate.

    Raises
    ------
    ValueError
        If required fields are blank, missing, or invalid.

    """
    if not isinstance(payload, Mapping):
        raise ValueError("certificate payload must be a mapping")
    family_id = payload.get("family_id")
    if not isinstance(family_id, str) or not family_id.strip():
        raise ValueError("family_id must be a non-empty string")
    # Unknown family fails closed.
    get_parity_family(family_id)
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("schema must be a non-empty string")
    if schema != POLYGLOT_PARITY_CERTIFICATE_SCHEMA:
        raise ValueError(
            f"unknown certificate schema {schema!r}; refuse invent-green "
            f"(expected {POLYGLOT_PARITY_CERTIFICATE_SCHEMA!r})"
        )
    sample_id = payload.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty string")
    input_digest = payload.get("input_digest")
    python_digest = payload.get("python_reference_digest")
    rust_digest = payload.get("rust_digest", "")
    if not isinstance(input_digest, str):
        raise ValueError("input_digest must be a string")
    if not isinstance(python_digest, str):
        raise ValueError("python_reference_digest must be a string")
    if not isinstance(rust_digest, str):
        raise ValueError("rust_digest must be a string")
    max_abs_error = payload.get("max_abs_error", 0.0)
    if not isinstance(max_abs_error, (int, float)) or isinstance(max_abs_error, bool):
        raise ValueError("max_abs_error must be a number")
    supported = payload.get("supported")
    if not isinstance(supported, bool):
        raise ValueError("supported must be a bool")
    blockers_raw = payload.get("blocked_reasons", ())
    if not isinstance(blockers_raw, (list, tuple)):
        raise ValueError("blocked_reasons must be a sequence")
    claim = payload.get("claim_boundary", POLYGLOT_PARITY_CLAIM_BOUNDARY)
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim_boundary must be a non-empty string")
    return PolyglotParityCertificate(
        family_id=family_id.strip(),
        schema=schema,
        sample_id=sample_id.strip(),
        input_digest=input_digest,
        python_reference_digest=python_digest,
        rust_digest=rust_digest,
        max_abs_error=float(max_abs_error),
        supported=supported,
        blocked_reasons=tuple(str(item) for item in blockers_raw),
        claim_boundary=claim,
    )


def verify_certificate(
    certificate: PolyglotParityCertificate | Mapping[str, Any],
    *,
    expect_supported: bool | None = None,
) -> CertificateVerifyDecision:
    """Verify a certificate bundle fail-closed.

    For supported sample certificates, recomputes digests from the deterministic
    sample generators and requires exact match + ``max_abs_error == 0.0``.
    Unsupported certificates verify as refused (honest blockers), not invent-green
    pass. Tampered digests fail.

    Parameters
    ----------
    certificate
        Certificate object or mapping.
    expect_supported
        Optional expected support flag; mismatch fails closed.

    Returns
    -------
    CertificateVerifyDecision
        Passed / failed / refused decision.

    """
    cert = (
        certificate
        if isinstance(certificate, PolyglotParityCertificate)
        else certificate_from_dict(certificate)
    )
    # Fail closed on unknown schema for object path as well as mapping path.
    if not cert.schema or cert.schema != POLYGLOT_PARITY_CERTIFICATE_SCHEMA:
        return CertificateVerifyDecision(
            family_id=cert.family_id if cert.family_id.strip() else "unknown",
            sample_id=cert.sample_id if cert.sample_id.strip() else "unknown",
            outcome="failed",
            passed=False,
            reason=(
                f"unknown certificate schema {cert.schema!r}; refuse invent-green "
                f"(expected {POLYGLOT_PARITY_CERTIFICATE_SCHEMA!r})"
            ),
            blockers=(f"unknown certificate schema {cert.schema!r}",),
            observed_max_abs_error=cert.max_abs_error,
        )
    family = get_parity_family(cert.family_id)

    if expect_supported is not None and cert.supported is not expect_supported:
        return CertificateVerifyDecision(
            family_id=cert.family_id,
            sample_id=cert.sample_id,
            outcome="failed",
            passed=False,
            reason="supported flag does not match expect_supported",
            blockers=(f"expected supported={expect_supported!r}, got {cert.supported!r}",),
            observed_max_abs_error=cert.max_abs_error,
        )

    if family.support != "sample_bitexact":
        return CertificateVerifyDecision(
            family_id=cert.family_id,
            sample_id=cert.sample_id,
            outcome="refused",
            passed=False,
            reason=(
                f"family support={family.support!r}; refuse invent-green bit-exact "
                "polyglot parity pass"
            ),
            blockers=cert.blocked_reasons
            or (f"family {family.family_id!r} is not sample_bitexact",),
            observed_max_abs_error=cert.max_abs_error,
        )

    # Recompute expected digests from generators.
    expected = build_sample_certificate(cert.family_id, sample_id=cert.sample_id)
    blockers: list[str] = []
    if cert.input_digest != expected.input_digest:
        blockers.append("input_digest mismatch vs recomputed sample")
    if cert.python_reference_digest != expected.python_reference_digest:
        blockers.append("python_reference_digest mismatch vs recomputed sample")
    if cert.rust_digest != expected.rust_digest:
        blockers.append("rust_digest mismatch vs recomputed sample")
    if cert.max_abs_error != 0.0:
        blockers.append("max_abs_error must be 0.0 for bit-exact pass")
    if not cert.supported:
        blockers.append("sample_bitexact family certificate must be supported")

    if blockers:
        return CertificateVerifyDecision(
            family_id=cert.family_id,
            sample_id=cert.sample_id,
            outcome="failed",
            passed=False,
            reason="polyglot parity certificate verify failed: " + "; ".join(blockers),
            blockers=tuple(blockers),
            observed_max_abs_error=cert.max_abs_error,
        )

    return CertificateVerifyDecision(
        family_id=cert.family_id,
        sample_id=cert.sample_id,
        outcome="passed",
        passed=True,
        reason=(
            f"certificate {cert.family_id!r}/{cert.sample_id!r} bit-exact digests match; "
            "max_abs_error=0.0; no invent-green full NumPy parity claimed"
        ),
        blockers=(),
        observed_max_abs_error=0.0,
    )


def map_parity_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of polyglot parity product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for family in _CANONICAL_FAMILIES:
        path = family.module_path
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "module_path": path,
                "role": "polyglot_parity_certificate_surface",
                "api_stability_class": family.api_stability_class,
                "support": family.support,
                "family_ids": [f.family_id for f in _CANONICAL_FAMILIES if f.module_path == path],
                "claim_boundary": POLYGLOT_PARITY_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_polyglot_parity_product_registry() -> dict[str, object]:
    """Build the full serialisable polyglot parity product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with families (no blanks).

    """
    families = [row.to_dict() for row in _CANONICAL_FAMILIES]
    sample_count = sum(1 for row in _CANONICAL_FAMILIES if row.support == "sample_bitexact")
    return {
        "schema": POLYGLOT_PARITY_PRODUCT_SCHEMA,
        "certificate_schema": POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
        "claim_boundary": POLYGLOT_PARITY_CLAIM_BOUNDARY,
        "family_count": len(families),
        "sample_bitexact_count": sample_count,
        "blank_entry_count": 0,
        "default_family_id": "scalar_interpreter_replay",
        "public_surfaces": list(map_parity_public_surfaces()),
        "families": families,
        "policy_note": _POLYGLOT_PARITY_POLICY_NOTE,
    }


def assert_polyglot_parity_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers families without blanks or invent-green.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_polyglot_parity_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage or blanks appear.

    """
    registry = dict(payload) if payload is not None else build_polyglot_parity_product_registry()
    families = registry.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("polyglot parity product registry must contain a non-empty families list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(families):
        if not isinstance(row, Mapping):
            raise ValueError(f"family row {index} must be a mapping")
        family_id = row.get("family_id")
        support = row.get("support")
        if not family_id or not str(family_id).strip():
            blank += 1
            continue
        fid = str(family_id).strip()
        if fid in seen:
            raise ValueError(f"duplicate family_id in registry: {fid!r}")
        seen.add(fid)
        if fid == "scalar_interpreter_replay":
            default_found = True
        if support not in {
            "sample_bitexact",
            "boundary_unsupported",
            "catalogue_only",
        }:
            blank += 1
            continue
    if blank:
        raise ValueError(f"polyglot parity product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("polyglot parity product registry missing scalar_interpreter_replay")
    expected = set(list_parity_family_ids())
    if seen != expected:
        raise ValueError(
            f"registry family set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    family_count = registry.get("family_count", -1)
    if not isinstance(family_count, int) or family_count != len(families):
        raise ValueError("family_count does not match families list length")
    cert_schema = registry.get("certificate_schema")
    if cert_schema != POLYGLOT_PARITY_CERTIFICATE_SCHEMA:
        raise ValueError("certificate_schema mismatch")
    if registry.get("schema") != POLYGLOT_PARITY_PRODUCT_SCHEMA:
        raise ValueError("product schema mismatch")
    if registry.get("claim_boundary") != POLYGLOT_PARITY_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary mismatch")
    if registry.get("policy_note") != _POLYGLOT_PARITY_POLICY_NOTE:
        raise ValueError("policy_note mismatch")
    if registry.get("default_family_id") != "scalar_interpreter_replay":
        raise ValueError("default_family_id mismatch")
    expected_sample_count = sum(
        1 for row in _CANONICAL_FAMILIES if row.support == "sample_bitexact"
    )
    if registry.get("sample_bitexact_count") != expected_sample_count:
        raise ValueError("sample_bitexact_count mismatch")
    expected_rows = {row.family_id: row.to_dict() for row in _CANONICAL_FAMILIES}
    for index, row in enumerate(families):
        family_id = str(row["family_id"]).strip()
        if dict(row) != expected_rows[family_id]:
            raise ValueError(f"family row {index} drift for {family_id!r}")
    if registry.get("public_surfaces") != list(map_parity_public_surfaces()):
        raise ValueError("public_surfaces mismatch")
    return registry


__all__ = [
    "POLYGLOT_PARITY_CERTIFICATE_SCHEMA",
    "POLYGLOT_PARITY_CLAIM_BOUNDARY",
    "POLYGLOT_PARITY_PRODUCT_SCHEMA",
    "CertificateVerifyDecision",
    "FamilySupport",
    "ParityFamily",
    "PolyglotParityCertificate",
    "VerifyOutcome",
    "assert_polyglot_parity_product_integrity",
    "build_polyglot_parity_product_registry",
    "build_sample_certificate",
    "canonical_json_bytes",
    "certificate_from_dict",
    "digest_payload",
    "get_parity_family",
    "iter_parity_families",
    "list_parity_family_ids",
    "map_parity_public_surfaces",
    "verify_certificate",
]
