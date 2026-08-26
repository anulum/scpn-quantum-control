# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for attested result packs (BL-48)
"""Real-surface tests for ``scpn_quantum_control.attested_result_pack``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scpn_quantum_control.attested_result_pack import (
    ATTESTED_RESULT_PACK_CLAIM_BOUNDARY,
    ATTESTED_RESULT_PACK_SCHEMA,
    AttestationVerdict,
    AttestedEnvelope,
    build_attestation_report,
    build_unsigned_envelope,
    canonical_content_digest,
    default_claim_axes,
    envelope_from_mapping,
    refuse_invent_green_hardware_attestation,
    verify_attested_envelope,
)


def _axes() -> dict[str, object]:
    return default_claim_axes()


def test_build_and_verify_unsigned_envelope() -> None:
    """Build and verify a digest-bound unsigned local envelope."""
    content = {"metric": 0.5, "n_shots": 0, "backend": "statevector"}
    env = build_unsigned_envelope(
        claim_id="claim.local.demo",
        claim_axes=_axes(),
        content=content,
    )
    assert env.schema == ATTESTED_RESULT_PACK_SCHEMA
    assert env.signature == ""
    assert len(env.content_digest) == 64
    verdict = verify_attested_envelope(env)
    assert verdict.status == "VERIFIED"
    assert verdict.expected_digest == env.content_digest
    assert verdict.claim_boundary == ATTESTED_RESULT_PACK_CLAIM_BOUNDARY


def test_stripped_missing_axes() -> None:
    """Classify an envelope with missing required axes as stripped."""
    env = build_unsigned_envelope(
        claim_id="claim.x",
        claim_axes={"claim_unit": "u"},  # incomplete
        content={"v": 1},
    )
    # build still digests, but verify must STRIPPED
    verdict = verify_attested_envelope(env)
    assert verdict.status == "STRIPPED"
    assert "missing" in verdict.reason.lower() or "axes" in verdict.reason.lower()


def test_forged_tampered_content() -> None:
    """Classify content changed after digesting as forged."""
    axes = _axes()
    env = build_unsigned_envelope(
        claim_id="claim.y",
        claim_axes=axes,
        content={"v": 1},
    )
    tampered = replace(env, content={"v": 999})
    verdict = verify_attested_envelope(tampered)
    assert verdict.status == "FORGED"
    assert verdict.expected_digest != verdict.observed_digest


def test_ungraded_blank_digest_and_require_signature() -> None:
    """Leave blank digests and missing required signatures ungraded."""
    axes = _axes()
    content = {"v": 1}
    env = build_unsigned_envelope(claim_id="claim.z", claim_axes=axes, content=content)
    blank = replace(env, content_digest="")
    verdict = verify_attested_envelope(blank)
    assert verdict.status == "UNGRADED"
    assert "digest" in verdict.reason.lower()

    signed_required = verify_attested_envelope(env, require_signature=True)
    assert signed_required.status == "UNGRADED"
    assert "signature" in signed_required.reason.lower()


def test_signature_present_without_keyring_is_ungraded() -> None:
    """Refuse to verify an opaque signature without a keyring."""
    env = build_unsigned_envelope(
        claim_id="claim.sig",
        claim_axes=_axes(),
        content={"v": 2},
    )
    tagged = replace(env, signature="opaque-token-not-verified")
    verdict = verify_attested_envelope(tagged)
    assert verdict.status == "UNGRADED"
    assert "keyring" in verdict.reason.lower() or "signature" in verdict.reason.lower()


def test_refuse_invent_green_hardware() -> None:
    """Keep hardware attestation ungraded with or without a digest."""
    no_digest = refuse_invent_green_hardware_attestation(
        claim_id="claim.hw",
        has_content_digest=False,
    )
    assert no_digest.status == "UNGRADED"
    with_digest = refuse_invent_green_hardware_attestation(
        claim_id="claim.hw",
        has_content_digest=True,
    )
    assert with_digest.status == "UNGRADED"
    assert "hardware" in with_digest.reason.lower()
    with pytest.raises(ValueError, match="claim_id"):
        refuse_invent_green_hardware_attestation(
            claim_id="  ",
            has_content_digest=False,
        )


def test_canonical_digest_stable_and_bound_to_axes() -> None:
    """Canonicalise mapping order while binding the claim axes."""
    content = {"a": 1, "b": 2}
    axes = _axes()
    d1 = canonical_content_digest(content, axes)
    d2 = canonical_content_digest({"b": 2, "a": 1}, axes)
    assert d1 == d2
    other_axes = default_claim_axes(backend_class="other")
    assert canonical_content_digest(content, other_axes) != d1


def test_verify_from_mapping_and_report() -> None:
    """Verify a mapping envelope and aggregate its verdict."""
    env = build_unsigned_envelope(
        claim_id="claim.map",
        claim_axes=_axes(),
        content={"x": True},
    )
    verdict = verify_attested_envelope(env.to_dict())
    assert verdict.status == "VERIFIED"
    report = build_attestation_report((verdict,))
    assert report["verdict_count"] == 1
    status_counts = report["status_counts"]
    assert isinstance(status_counts, dict)
    assert status_counts["VERIFIED"] == 1
    assert report["schema"] == ATTESTED_RESULT_PACK_SCHEMA


def test_default_claim_axes_validation() -> None:
    """Build complete default axes and reject blank axis labels."""
    axes = default_claim_axes()
    assert axes["claim_unit"]
    assert axes["honesty_axes"]
    with pytest.raises(ValueError, match="claim_unit"):
        default_claim_axes(claim_unit="")
    with pytest.raises(ValueError, match="backend_class"):
        default_claim_axes(backend_class="  ")
    with pytest.raises(ValueError, match="honesty_axes"):
        default_claim_axes(honesty_axes=["  ", ""])


def test_envelope_and_verdict_validation() -> None:
    """Enforce envelope and verdict construction invariants."""
    with pytest.raises(ValueError, match="claim_id"):
        AttestedEnvelope(
            claim_id="",
            claim_axes=_axes(),
            content={"v": 1},
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="content_digest"):
        AttestedEnvelope(
            claim_id="c",
            claim_axes=_axes(),
            content={"v": 1},
            content_digest="zz",
        )
    with pytest.raises(TypeError, match="claim_axes"):
        AttestedEnvelope(
            claim_id="c",
            claim_axes=["not", "a", "mapping"],  # type: ignore[arg-type]
            content={"v": 1},
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="unknown attestation status"):
        AttestationVerdict(
            status="OK",  # type: ignore[arg-type]
            reason="r",
            expected_digest="",
            observed_digest="",
            claim_id="c",
        )
    with pytest.raises(ValueError, match="reason"):
        AttestationVerdict(
            status="VERIFIED",
            reason="",
            expected_digest="",
            observed_digest="",
            claim_id="c",
        )
    with pytest.raises(ValueError, match="claim_id"):
        AttestationVerdict(
            status="VERIFIED",
            reason="r",
            expected_digest="",
            observed_digest="",
            claim_id="",
        )


def test_build_unsigned_requires_claim_id() -> None:
    """Reject a blank claim identifier when building an envelope."""
    with pytest.raises(ValueError, match="claim_id"):
        build_unsigned_envelope(claim_id="  ", claim_axes=_axes(), content={"v": 1})


def test_canonical_digest_rejects_non_mapping_and_non_json() -> None:
    """Reject non-mapping inputs and non-canonical JSON values."""
    with pytest.raises(TypeError, match="content"):
        canonical_content_digest("x", _axes())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="claim_axes"):
        canonical_content_digest({"a": 1}, "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON"):
        canonical_content_digest({"a": {1, 2}}, _axes())


def test_verify_type_errors_and_undigestable_mapping() -> None:
    """Reject invalid envelope types and leave NaN content ungraded."""
    with pytest.raises(TypeError, match="envelope"):
        verify_attested_envelope(123)  # type: ignore[arg-type]
    # undigestable content via mapping path with bad nested type after construction
    # is hard; exercise undigestable path by forging AttestedEnvelope with bad content
    # using object that fails json - use float('nan') which allow_nan=False rejects
    env = AttestedEnvelope(
        claim_id="c",
        claim_axes=_axes(),
        content={"v": float("nan")},
        content_digest="a" * 64,
    )
    verdict = verify_attested_envelope(env)
    assert verdict.status == "UNGRADED"
    assert "digestable" in verdict.reason.lower() or "json" in verdict.reason.lower()


def test_build_attestation_report_type_checks() -> None:
    """Reject invalid report containers and verdict elements."""
    with pytest.raises(TypeError, match="list or tuple"):
        build_attestation_report("nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AttestationVerdict"):
        build_attestation_report([1])  # type: ignore[list-item]


def test_envelope_from_mapping_round_trip() -> None:
    """Round-trip nested mapping content through envelope construction."""
    env = build_unsigned_envelope(
        claim_id="claim.round",
        claim_axes=_axes(),
        content={"k": [1, 2], "nested": {"a": 1}},
    )
    restored = envelope_from_mapping(env.to_dict())
    assert restored.claim_id == env.claim_id
    assert restored.content_digest == env.content_digest
    assert restored.content["nested"] == {"a": 1}
    assert verify_attested_envelope(restored).status == "VERIFIED"
    with pytest.raises(TypeError, match="mapping"):
        envelope_from_mapping("x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="claim_axes"):
        envelope_from_mapping({"claim_id": "c", "claim_axes": [], "content": {}})


def test_stripped_empty_honesty_axes_list() -> None:
    """Classify an empty honesty-axis list as stripped."""
    axes = {
        "claim_unit": "u",
        "honesty_axes": [],
        "backend_class": "simulator_local",
    }
    env = build_unsigned_envelope(claim_id="claim.empty", claim_axes=axes, content={"v": 1})
    assert verify_attested_envelope(env).status == "STRIPPED"


def test_stripped_none_and_blank_string_axes() -> None:
    """Classify null and blank required axis values as stripped."""
    axes_none = {
        "claim_unit": None,
        "honesty_axes": ["inputs_digest"],
        "backend_class": "simulator_local",
    }
    env_none = build_unsigned_envelope(
        claim_id="claim.none",
        claim_axes=axes_none,
        content={"v": 1},
    )
    assert verify_attested_envelope(env_none).status == "STRIPPED"

    axes_blank = {
        "claim_unit": "u",
        "honesty_axes": ["inputs_digest"],
        "backend_class": "   ",
    }
    env_blank = build_unsigned_envelope(
        claim_id="claim.blank",
        claim_axes=axes_blank,
        content={"v": 1},
    )
    assert verify_attested_envelope(env_blank).status == "STRIPPED"


def test_content_must_be_mapping() -> None:
    """Reject non-mapping envelope content."""
    with pytest.raises(TypeError, match="content"):
        AttestedEnvelope(
            claim_id="c",
            claim_axes=_axes(),
            content=["not", "mapping"],  # type: ignore[arg-type]
            content_digest="a" * 64,
        )


def test_verify_mapping_bad_axes_type() -> None:
    """Reject mapping envelopes whose claim axes are not mappings."""
    with pytest.raises(TypeError, match="claim_axes and content"):
        verify_attested_envelope(
            {
                "claim_id": "c",
                "claim_axes": "bad",
                "content": {},
                "content_digest": "a" * 64,
            }
        )


def test_verdict_to_dict() -> None:
    """Serialise every verdict field and the shared claim boundary."""
    v = AttestationVerdict(
        status="VERIFIED",
        reason="ok",
        expected_digest="a" * 64,
        observed_digest="a" * 64,
        claim_id="c",
    )
    payload = v.to_dict()
    assert payload["status"] == "VERIFIED"
    assert payload["claim_boundary"] == ATTESTED_RESULT_PACK_CLAIM_BOUNDARY
