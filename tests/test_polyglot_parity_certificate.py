# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for polyglot parity certificates
"""Real-surface tests for ``scpn_quantum_control.polyglot_parity_certificate``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.polyglot_parity_certificate as polyglot_parity_certificate
from scpn_quantum_control.polyglot_parity_certificate import (
    POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
    POLYGLOT_PARITY_CLAIM_BOUNDARY,
    POLYGLOT_PARITY_PRODUCT_SCHEMA,
    CertificateVerifyDecision,
    ParityFamily,
    PolyglotParityCertificate,
    assert_polyglot_parity_product_integrity,
    build_polyglot_parity_product_registry,
    build_sample_certificate,
    canonical_json_bytes,
    certificate_from_dict,
    digest_payload,
    get_parity_family,
    iter_parity_families,
    list_parity_family_ids,
    map_parity_public_surfaces,
    verify_certificate,
)


def test_list_families_and_filters() -> None:
    """Keep family ordering deterministic and support filtering exact."""
    ids = list_parity_family_ids()
    assert "scalar_interpreter_replay" in ids
    assert "value_and_gradient_replay" in ids
    assert "elementwise_primitive_parity" in ids
    assert ids == list_parity_family_ids()
    sample = iter_parity_families(support="sample_bitexact")
    assert sample
    assert all(row.support == "sample_bitexact" for row in sample)


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve declared families while refusing blank and unknown identifiers."""
    family = get_parity_family("scalar_interpreter_replay")
    assert family.api_stability_class == "experimental_workbench"
    assert family.claim_boundary == POLYGLOT_PARITY_CLAIM_BOUNDARY
    with pytest.raises(ValueError, match="non-empty"):
        get_parity_family("  ")
    with pytest.raises(ValueError, match="unknown family_id"):
        get_parity_family("not_a_family")


def test_build_and_verify_sample_bitexact() -> None:
    """Build and verify a supported deterministic bit-exact sample."""
    cert = build_sample_certificate("scalar_interpreter_replay")
    assert cert.supported is True
    assert cert.max_abs_error == 0.0
    assert len(cert.input_digest) == 64
    assert len(cert.python_reference_digest) == 64
    assert len(cert.rust_digest) == 64
    assert cert.schema == POLYGLOT_PARITY_CERTIFICATE_SCHEMA
    decision = verify_certificate(cert)
    assert decision.passed is True
    assert decision.outcome == "passed"
    assert decision.observed_max_abs_error == 0.0
    assert "bit-exact" in decision.reason


def test_build_boundary_and_catalogue_refuse() -> None:
    """Represent boundary and catalogue-only families as honest refusals."""
    boundary = build_sample_certificate("elementwise_primitive_parity")
    assert boundary.supported is False
    assert boundary.blocked_reasons
    refused = verify_certificate(boundary)
    assert refused.passed is False
    assert refused.outcome == "refused"

    catalogue = build_sample_certificate("spectral_bounds_parity")
    assert catalogue.supported is False
    cat_decision = verify_certificate(catalogue)
    assert cat_decision.passed is False
    assert cat_decision.outcome == "refused"


def test_verify_detects_tamper() -> None:
    """Fail verification when a certificate digest is tampered."""
    cert = build_sample_certificate("value_and_gradient_replay")
    tampered = PolyglotParityCertificate(
        family_id=cert.family_id,
        schema=cert.schema,
        sample_id=cert.sample_id,
        input_digest=cert.input_digest,
        python_reference_digest=cert.python_reference_digest,
        rust_digest="0" * 64,
        max_abs_error=0.0,
        supported=True,
        blocked_reasons=(),
    )
    decision = verify_certificate(tampered)
    assert decision.passed is False
    assert decision.outcome == "failed"
    assert any("rust_digest" in item for item in decision.blockers)


def test_verify_expect_supported_mismatch() -> None:
    """Fail when the certificate support flag violates caller expectation."""
    cert = build_sample_certificate("scalar_interpreter_replay")
    decision = verify_certificate(cert, expect_supported=False)
    assert decision.passed is False
    assert decision.outcome == "failed"


def test_certificate_from_dict_and_round_trip() -> None:
    """Round-trip certificate mappings through validation and verification."""
    cert = build_sample_certificate("registry_metadata_mirror", sample_id="sample-1")
    rebuilt = certificate_from_dict(cert.to_dict())
    assert rebuilt.family_id == cert.family_id
    assert rebuilt.sample_id == "sample-1"
    assert verify_certificate(rebuilt).passed is True
    # mapping path
    assert verify_certificate(cert.to_dict()).passed is True


def test_certificate_from_dict_fail_closed() -> None:
    """Reject malformed, unknown, and ill-typed certificate mappings."""
    with pytest.raises(ValueError, match="mapping"):
        certificate_from_dict(cast(Any, "nope"))
    with pytest.raises(ValueError, match="family_id"):
        certificate_from_dict({"family_id": ""})
    with pytest.raises(ValueError, match="unknown family_id"):
        certificate_from_dict(
            {
                "family_id": "ghost",
                "schema": POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
                "sample_id": "s",
                "input_digest": "a" * 64,
                "python_reference_digest": "b" * 64,
                "rust_digest": "c" * 64,
                "max_abs_error": 0.0,
                "supported": True,
                "blocked_reasons": [],
            }
        )
    with pytest.raises(ValueError, match="unknown certificate schema"):
        certificate_from_dict(
            {
                "family_id": "scalar_interpreter_replay",
                "schema": "ghost.v0",
                "sample_id": "s",
                "input_digest": "a" * 64,
                "python_reference_digest": "b" * 64,
                "rust_digest": "c" * 64,
                "max_abs_error": 0.0,
                "supported": True,
                "blocked_reasons": [],
            }
        )
    with pytest.raises(ValueError, match="sample_id"):
        certificate_from_dict(
            {
                "family_id": "scalar_interpreter_replay",
                "schema": POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
                "sample_id": "",
                "input_digest": "a" * 64,
                "python_reference_digest": "b" * 64,
                "rust_digest": "",
                "max_abs_error": 0.0,
                "supported": False,
                "blocked_reasons": ["x"],
            }
        )
    with pytest.raises(ValueError, match="supported must be a bool"):
        certificate_from_dict(
            {
                "family_id": "scalar_interpreter_replay",
                "schema": POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
                "sample_id": "s",
                "input_digest": "a" * 64,
                "python_reference_digest": "b" * 64,
                "rust_digest": "",
                "max_abs_error": 0.0,
                "supported": "yes",
                "blocked_reasons": ["x"],
            }
        )


def test_registry_and_integrity() -> None:
    """Build and validate both explicit and default product registries."""
    assert POLYGLOT_PARITY_CERTIFICATE_SCHEMA == "polyglot_parity_certificate.v2"
    assert POLYGLOT_PARITY_PRODUCT_SCHEMA == "polyglot_parity_certificate_product.v2"
    registry = build_polyglot_parity_product_registry()
    assert registry["schema"] == POLYGLOT_PARITY_PRODUCT_SCHEMA
    assert registry["certificate_schema"] == POLYGLOT_PARITY_CERTIFICATE_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_family_id"] == "scalar_interpreter_replay"
    validated = assert_polyglot_parity_product_integrity(registry)
    assert validated["family_count"] == len(list_parity_family_ids())
    assert assert_polyglot_parity_product_integrity()["blank_entry_count"] == 0


def test_public_surfaces() -> None:
    """Map every owning module to the governed certificate surface role."""
    surfaces = map_parity_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.program_ad_rust_bridge" in paths
    for row in surfaces:
        assert row["role"] == "polyglot_parity_certificate_surface"


def test_integrity_rejects_drift() -> None:
    """Reject extra family identifiers and empty registry inventories."""
    registry = build_polyglot_parity_product_registry()
    families = cast(list[dict[str, object]], registry["families"])
    broken = dict(registry)
    broken["families"] = families + [
        {
            "family_id": "ghost",
            "title": "t",
            "summary": "s",
            "support": "sample_bitexact",
            "module_path": "m",
            "api_stability_class": "experimental_workbench",
            "as_of": "2026-07-24",
            "claim_boundary": POLYGLOT_PARITY_CLAIM_BOUNDARY,
        }
    ]
    broken["family_count"] = len(cast(list[object], broken["families"]))
    with pytest.raises(ValueError, match="drift"):
        assert_polyglot_parity_product_integrity(broken)

    empty: dict[str, object] = {"families": [], "blank_entry_count": 0, "family_count": 0}
    with pytest.raises(ValueError, match="non-empty families"):
        assert_polyglot_parity_product_integrity(empty)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed rows, blank fields, duplicates, counts, and schemas."""
    registry = build_polyglot_parity_product_registry()
    families = cast(list[dict[str, object]], registry["families"])

    non_map = dict(registry)
    non_map["families"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_polyglot_parity_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in families]
    rows[0]["family_id"] = "  "
    blank_id["families"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_polyglot_parity_product_integrity(blank_id)

    bad_support = dict(registry)
    srows = [dict(row) for row in families]
    srows[1]["support"] = "nope"
    bad_support["families"] = srows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_polyglot_parity_product_integrity(bad_support)

    no_default = dict(registry)
    renamed = [dict(row) for row in families]
    for row in renamed:
        if row.get("family_id") == "scalar_interpreter_replay":
            row["family_id"] = "renamed"
    no_default["families"] = renamed
    with pytest.raises(ValueError, match="missing scalar_interpreter_replay|drift"):
        assert_polyglot_parity_product_integrity(no_default)

    dup = dict(registry)
    drows = [dict(row) for row in families]
    drows.append(dict(drows[0]))
    dup["families"] = drows
    dup["family_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate family_id"):
        assert_polyglot_parity_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_polyglot_parity_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["family_count"] = 0
    with pytest.raises(ValueError, match="family_count"):
        assert_polyglot_parity_product_integrity(count_mismatch)

    bad_schema = dict(registry)
    bad_schema["certificate_schema"] = "ghost"
    with pytest.raises(ValueError, match="certificate_schema"):
        assert_polyglot_parity_product_integrity(bad_schema)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "polyglot_parity_certificate_product.v1", "product schema"),
        ("claim_boundary", "drifted claim", "claim_boundary"),
        ("policy_note", "drifted policy", "policy_note"),
        ("default_family_id", "value_and_gradient_replay", "default_family_id"),
        ("sample_bitexact_count", 0, "sample_bitexact_count"),
    ],
)
def test_integrity_rejects_governed_metadata_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject stale or altered top-level registry contracts."""
    registry = build_polyglot_parity_product_registry()
    registry[field] = value
    with pytest.raises(ValueError, match=message):
        assert_polyglot_parity_product_integrity(registry)


def test_integrity_rejects_canonical_row_and_surface_drift() -> None:
    """Reject altered family metadata and public-surface inventories."""
    registry = build_polyglot_parity_product_registry()
    family_drift = dict(registry)
    families = [dict(row) for row in cast(list[dict[str, object]], registry["families"])]
    families[0]["claim_boundary"] = "drifted claim"
    family_drift["families"] = families
    with pytest.raises(ValueError, match="family row 0 drift"):
        assert_polyglot_parity_product_integrity(family_drift)

    surface_drift = dict(registry)
    surface_drift["public_surfaces"] = []
    with pytest.raises(ValueError, match="public_surfaces"):
        assert_polyglot_parity_product_integrity(surface_drift)


def test_module_exports() -> None:
    """Keep the documented catalogue, builder, and verifier APIs exported."""
    assert "build_sample_certificate" in polyglot_parity_certificate.__all__
    assert "verify_certificate" in polyglot_parity_certificate.__all__
    assert "list_parity_family_ids" in polyglot_parity_certificate.__all__


def test_family_validation() -> None:
    """Reject every invalid family-catalogue invariant at construction."""
    base: dict[str, Any] = {
        "family_id": "x",
        "title": "t",
        "summary": "s",
        "support": "sample_bitexact",
        "module_path": "m",
    }
    assert ParityFamily(**base).family_id == "x"
    with pytest.raises(ValueError, match="family_id"):
        ParityFamily(**{**base, "family_id": ""})
    with pytest.raises(ValueError, match="title"):
        ParityFamily(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        ParityFamily(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="support"):
        ParityFamily(**{**base, "support": cast(Any, "nope")})
    with pytest.raises(ValueError, match="module_path"):
        ParityFamily(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="api_stability_class"):
        ParityFamily(**{**base, "api_stability_class": ""})
    with pytest.raises(ValueError, match="as_of"):
        ParityFamily(**{**base, "as_of": ""})
    with pytest.raises(ValueError, match="claim_boundary"):
        ParityFamily(**{**base, "claim_boundary": "drifted claim"})


def test_certificate_invariants() -> None:
    """Reject inconsistent supported and blocked certificate records."""
    good: dict[str, Any] = dict(
        family_id="scalar_interpreter_replay",
        schema=POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
        sample_id="s",
        input_digest="a" * 64,
        python_reference_digest="b" * 64,
        rust_digest="c" * 64,
        max_abs_error=0.0,
        supported=True,
        blocked_reasons=(),
    )
    assert PolyglotParityCertificate(**good).supported is True
    with pytest.raises(ValueError, match="family_id"):
        PolyglotParityCertificate(**{**good, "family_id": ""})
    with pytest.raises(ValueError, match="schema"):
        PolyglotParityCertificate(**{**good, "schema": ""})
    with pytest.raises(ValueError, match="sample_id"):
        PolyglotParityCertificate(**{**good, "sample_id": ""})
    with pytest.raises(ValueError, match="input_digest"):
        PolyglotParityCertificate(**{**good, "input_digest": "short"})
    with pytest.raises(ValueError, match="input_digest"):
        PolyglotParityCertificate(**{**good, "input_digest": "g" * 64})
    with pytest.raises(ValueError, match="rust_digest"):
        PolyglotParityCertificate(**{**good, "rust_digest": "short"})
    with pytest.raises(ValueError, match="max_abs_error"):
        PolyglotParityCertificate(**{**good, "max_abs_error": -1.0})
    with pytest.raises(ValueError, match="cannot list blockers"):
        PolyglotParityCertificate(**{**good, "blocked_reasons": ("x",)})
    with pytest.raises(ValueError, match="require rust_digest"):
        PolyglotParityCertificate(**{**good, "rust_digest": ""})
    with pytest.raises(ValueError, match="max_abs_error == 0.0"):
        PolyglotParityCertificate(**{**good, "max_abs_error": 1e-9})
    with pytest.raises(ValueError, match="require blockers"):
        PolyglotParityCertificate(
            **{
                **good,
                "supported": False,
                "rust_digest": "",
                "blocked_reasons": (),
            }
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        PolyglotParityCertificate(**{**good, "claim_boundary": "drifted claim"})


def test_decision_invariants() -> None:
    """Reject malformed verification outcomes and decision metadata."""
    with pytest.raises(ValueError, match="family_id"):
        CertificateVerifyDecision(
            family_id="",
            sample_id="s",
            outcome="refused",
            passed=False,
            reason="r",
            blockers=("b",),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="outcome"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome=cast(Any, "nope"),
            passed=False,
            reason="r",
            blockers=("b",),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="require blockers"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="failed",
            passed=False,
            reason="r",
            blockers=(),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="passed",
            passed=True,
            reason="r",
            blockers=("b",),
            observed_max_abs_error=0.0,
        )
    ok = CertificateVerifyDecision(
        family_id="f",
        sample_id="s",
        outcome="passed",
        passed=True,
        reason="r",
        blockers=(),
        observed_max_abs_error=0.0,
    )
    assert ok.to_dict()["passed"] is True
    with pytest.raises(ValueError, match="claim_boundary"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="passed",
            passed=True,
            reason="r",
            blockers=(),
            observed_max_abs_error=0.0,
            claim_boundary="drifted claim",
        )


def test_digest_and_canonical() -> None:
    """Canonicalise mappings deterministically and digest their exact bytes."""
    payload = {"b": 2, "a": 1}
    digest = digest_payload(payload)
    assert len(digest) == 64
    assert digest == digest_payload({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="mapping"):
        canonical_json_bytes(cast(Any, [1, 2]))


def test_build_sample_id_blank() -> None:
    """Reject blank sample identifiers before certificate construction."""
    with pytest.raises(ValueError, match="sample_id"):
        build_sample_certificate("scalar_interpreter_replay", sample_id="  ")


def test_catalogue_map_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise blank, duplicate, and empty catalogue construction guards."""
    from scpn_quantum_control import polyglot_parity_certificate as mod

    good = get_parity_family("scalar_interpreter_replay")
    blank = ParityFamily(
        family_id="tmp",
        title="t",
        summary="s",
        support="sample_bitexact",
        module_path="m",
    )
    object.__setattr__(blank, "family_id", "  ")
    monkeypatch.setattr(mod, "_CANONICAL_FAMILIES", (blank,))
    with pytest.raises(RuntimeError, match="blank family_id"):
        mod._catalogue_map()

    a = get_parity_family("scalar_interpreter_replay")
    monkeypatch.setattr(mod, "_CANONICAL_FAMILIES", (a, a))
    with pytest.raises(RuntimeError, match="duplicate family_id"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_FAMILIES", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_FAMILIES", (good,))
    assert mod._catalogue_map()[good.family_id].family_id == good.family_id


def test_family_to_dict() -> None:
    """Serialise family records without losing policy or provenance fields."""
    family = get_parity_family("linalg_primitive_parity")
    payload = family.to_dict()
    assert payload["support"] == "boundary_unsupported"
    assert payload["family_id"] == "linalg_primitive_parity"


def test_certificate_from_dict_type_edges() -> None:
    """Reject certificate fields with invalid scalar and sequence types."""
    base = {
        "family_id": "scalar_interpreter_replay",
        "schema": POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
        "sample_id": "s",
        "input_digest": "a" * 64,
        "python_reference_digest": "b" * 64,
        "rust_digest": "c" * 64,
        "max_abs_error": 0.0,
        "supported": True,
        "blocked_reasons": [],
    }
    with pytest.raises(ValueError, match="schema must be a non-empty string"):
        certificate_from_dict({**base, "schema": 1})
    with pytest.raises(ValueError, match="input_digest must be a string"):
        certificate_from_dict({**base, "input_digest": 1})
    with pytest.raises(ValueError, match="python_reference_digest must be a string"):
        certificate_from_dict({**base, "python_reference_digest": 1})
    with pytest.raises(ValueError, match="rust_digest must be a string"):
        certificate_from_dict({**base, "rust_digest": 1})
    with pytest.raises(ValueError, match="max_abs_error must be a number"):
        certificate_from_dict({**base, "max_abs_error": True})
    with pytest.raises(ValueError, match="blocked_reasons must be a sequence"):
        certificate_from_dict({**base, "blocked_reasons": "x"})
    with pytest.raises(ValueError, match="claim_boundary must be a non-empty string"):
        certificate_from_dict({**base, "claim_boundary": ""})
    with pytest.raises(ValueError, match="claim_boundary"):
        certificate_from_dict({**base, "claim_boundary": "drifted claim"})
    with pytest.raises(ValueError, match="unknown certificate schema"):
        certificate_from_dict({**base, "schema": "polyglot_parity_certificate.v1"})


def test_certificate_blank_blockers_entry() -> None:
    """Reject blank blocker text in unsupported certificate mappings."""
    with pytest.raises(ValueError, match="blocked_reasons entries"):
        PolyglotParityCertificate(
            family_id="scalar_interpreter_replay",
            schema=POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
            sample_id="s",
            input_digest="a" * 64,
            python_reference_digest="b" * 64,
            rust_digest="",
            max_abs_error=0.0,
            supported=False,
            blocked_reasons=("ok", "  "),
        )


def test_decision_more_invariants() -> None:
    """Cover refused, passed, and blocker consistency invariants."""
    with pytest.raises(ValueError, match="sample_id"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="",
            outcome="refused",
            passed=False,
            reason="r",
            blockers=("b",),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="reason"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="refused",
            passed=False,
            reason="",
            blockers=("b",),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="must use outcome=passed"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="failed",
            passed=True,
            reason="r",
            blockers=(),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="cannot use outcome=passed"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="passed",
            passed=False,
            reason="r",
            blockers=("b",),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="blockers entries"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="failed",
            passed=False,
            reason="r",
            blockers=("ok", ""),
            observed_max_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="observed_max_abs_error"):
        CertificateVerifyDecision(
            family_id="f",
            sample_id="s",
            outcome="failed",
            passed=False,
            reason="r",
            blockers=("b",),
            observed_max_abs_error=-0.1,
        )


def test_certificate_object_rejects_unknown_schema() -> None:
    """Construction fail-closed for unknown schema (object path)."""
    good = build_sample_certificate("scalar_interpreter_replay")
    with pytest.raises(ValueError, match="unknown certificate schema"):
        PolyglotParityCertificate(
            family_id=good.family_id,
            schema="evil.v0",
            sample_id=good.sample_id,
            input_digest=good.input_digest,
            python_reference_digest=good.python_reference_digest,
            rust_digest=good.rust_digest,
            max_abs_error=0.0,
            supported=True,
            blocked_reasons=(),
        )
    with pytest.raises(ValueError, match="unknown certificate schema"):
        PolyglotParityCertificate(
            family_id=good.family_id,
            schema="ghost.v0",
            sample_id=good.sample_id,
            input_digest=good.input_digest,
            python_reference_digest=good.python_reference_digest,
            rust_digest=good.rust_digest,
            max_abs_error=0.0,
            supported=True,
            blocked_reasons=(),
        )


def test_verify_object_path_unknown_schema_fail_closed() -> None:
    """verify_certificate must fail closed when object schema is mutated/wrong.

    Reproduces skeptic gap: build_sample then re-wrap with schema='evil.v0'
    must not return passed=True even if digests match.
    """
    good = build_sample_certificate("scalar_interpreter_replay")
    # Bypass __post_init__ to simulate a tampered/deserialised object that
    # reached verify with a wrong schema (defense-in-depth path).
    evil = PolyglotParityCertificate(
        family_id=good.family_id,
        schema=POLYGLOT_PARITY_CERTIFICATE_SCHEMA,
        sample_id=good.sample_id,
        input_digest=good.input_digest,
        python_reference_digest=good.python_reference_digest,
        rust_digest=good.rust_digest,
        max_abs_error=0.0,
        supported=True,
        blocked_reasons=(),
    )
    object.__setattr__(evil, "schema", "evil.v0")
    decision = verify_certificate(evil)
    assert decision.passed is False
    assert decision.outcome == "failed"
    assert any("schema" in item.lower() for item in decision.blockers)
    assert "evil.v0" in decision.reason

    # Mapping path still refuses unknown schema (already covered; re-assert).
    mapping = good.to_dict()
    mapping["schema"] = "ghost.v0"
    with pytest.raises(ValueError, match="unknown certificate schema"):
        certificate_from_dict(mapping)
    with pytest.raises(ValueError, match="unknown certificate schema"):
        verify_certificate(mapping)


def test_verify_input_and_error_paths() -> None:
    """Exercise verifier mapping input and recomputed mismatch paths."""
    cert = build_sample_certificate("scalar_interpreter_replay")
    bad_input = PolyglotParityCertificate(
        family_id=cert.family_id,
        schema=cert.schema,
        sample_id=cert.sample_id,
        input_digest="f" * 64,
        python_reference_digest=cert.python_reference_digest,
        rust_digest=cert.rust_digest,
        max_abs_error=0.0,
        supported=True,
        blocked_reasons=(),
    )
    d1 = verify_certificate(bad_input)
    assert d1.passed is False
    assert any("input_digest" in item for item in d1.blockers)

    bad_py = PolyglotParityCertificate(
        family_id=cert.family_id,
        schema=cert.schema,
        sample_id=cert.sample_id,
        input_digest=cert.input_digest,
        python_reference_digest="e" * 64,
        rust_digest=cert.rust_digest,
        max_abs_error=0.0,
        supported=True,
        blocked_reasons=(),
    )
    d2 = verify_certificate(bad_py)
    assert d2.passed is False
    assert any("python_reference" in item for item in d2.blockers)


def test_iter_parity_families_without_filter_returns_full_catalogue() -> None:
    """Unfiltered family iter returns every catalogue row."""
    rows = iter_parity_families()
    assert len(rows) == len(list_parity_family_ids())
    assert {row.family_id for row in rows} == set(list_parity_family_ids())


def test_verify_rejects_nonzero_error_and_unsupported_on_bitexact_family() -> None:
    """sample_bitexact family with non-zero error and supported=False fails closed."""
    cert = build_sample_certificate("scalar_interpreter_replay")
    bad = PolyglotParityCertificate(
        family_id=cert.family_id,
        schema=cert.schema,
        sample_id=cert.sample_id,
        input_digest=cert.input_digest,
        python_reference_digest=cert.python_reference_digest,
        rust_digest=cert.rust_digest,
        max_abs_error=1e-6,
        supported=False,
        blocked_reasons=("forced unsupported for verify path",),
    )
    decision = verify_certificate(bad)
    assert decision.passed is False
    assert decision.outcome == "failed"
    assert any("max_abs_error" in item for item in decision.blockers)
    assert any("must be supported" in item for item in decision.blockers)
