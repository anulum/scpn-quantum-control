# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for WS-1 attestation-mode result-pack sealing
"""Tests for studio/result_pack_seal.py — sealing a hardware result pack.

Covers the claim-unit build from a real committed pack, the provider-attestation
guard, and the end-to-end attestation-mode seal verifying through the platform
verdict with QUANTUM's ML-DSA key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform.seal", reason="studio extra not installed")

from scpn_quantum_control.crypto.ml_dsa_seal import MLDSASigner  # noqa: E402
from scpn_quantum_control.studio.result_pack_seal import (  # noqa: E402
    DEFAULT_CLAIM_STATUS,
    DEFAULT_EVIDENCE_KIND,
    build_provider_attestation,
    build_result_pack_unit,
    seal_result_pack,
)
from scpn_quantum_control.studio.verbs import HARDWARE_RESULT_PACK_SCHEMA  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO_ROOT / "data" / "hardware_result_packs" / "manifest.json"
_SEED = bytes(range(32))
_RAW_DIGEST = "sha256:" + "11" * 32
_CIRCUIT_DIGEST = "sha256:" + "22" * 32
_GRADER = {"name": "honesty-bridge", "version": "0.8.0"}


def _real_pack() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))["packs"][0]


def _signer() -> MLDSASigner:
    return MLDSASigner.generate("scpn-quantum-control:qpu", seed=_SEED)


def _attestation() -> dict[str, str]:
    return build_provider_attestation(
        provider="ibm", result_pack_digest=_RAW_DIGEST, provider_sig="cHJvdmlkZXItc2ln"
    )


# ── unit construction from a real pack ─────────────────────────────────


def test_unit_carries_real_pack_provenance() -> None:
    """The unit binds the real pack's schema, digests, and provenance boundary."""
    pack = _real_pack()
    unit = build_result_pack_unit(
        pack, raw_results_digest=_RAW_DIGEST, circuit_digest=_CIRCUIT_DIGEST
    )
    assert unit["schema"] == HARDWARE_RESULT_PACK_SCHEMA
    assert unit["raw_results_digest"] == _RAW_DIGEST
    assert unit["circuit_digest"] == _CIRCUIT_DIGEST
    assert unit["evidence_kind"] == DEFAULT_EVIDENCE_KIND
    assert unit["claim_status"] == DEFAULT_CLAIM_STATUS
    assert unit["provenance"]["id"] == pack["id"]
    assert unit["provenance"]["backend"] == pack["backend"]
    # The non_claims boundary travels into the signed unit.
    assert "non_claims" in unit["provenance"]


def test_unit_omits_circuit_digest_when_absent() -> None:
    """A pack without a compiled-circuit link seals without the field, not with null."""
    unit = build_result_pack_unit(_real_pack(), raw_results_digest=_RAW_DIGEST)
    assert "circuit_digest" not in unit


def test_unit_honours_explicit_honesty_axes() -> None:
    """The producer may declare stronger/weaker axes than the conservative default."""
    unit = build_result_pack_unit(
        _real_pack(),
        raw_results_digest=_RAW_DIGEST,
        evidence_kind="noise-limited",
        claim_status="bounded-model",
    )
    assert unit["evidence_kind"] == "noise-limited"
    assert unit["claim_status"] == "bounded-model"


def test_unit_requires_pack_id() -> None:
    """A pack with no id cannot be sealed — there is nothing to address."""
    with pytest.raises(ValueError, match="non-empty id"):
        build_result_pack_unit({"backend": "ibm"}, raw_results_digest=_RAW_DIGEST)


def test_unit_requires_raw_results_digest() -> None:
    """An empty raw-results digest is refused — the attestation has nothing to bind."""
    with pytest.raises(ValueError, match="raw_results_digest"):
        build_result_pack_unit(_real_pack(), raw_results_digest="   ")


# ── provider attestation guard ─────────────────────────────────────────


def test_provider_attestation_round_trips() -> None:
    """A complete attestation carries the three load-bearing fields."""
    att = build_provider_attestation(
        provider="ibm", result_pack_digest=_RAW_DIGEST, provider_sig="c2ln"
    )
    assert att == {"provider": "ibm", "result_pack_digest": _RAW_DIGEST, "provider_sig": "c2ln"}


@pytest.mark.parametrize(
    ("provider", "digest", "sig"),
    [
        ("", _RAW_DIGEST, "c2ln"),
        ("ibm", "  ", "c2ln"),
        ("ibm", _RAW_DIGEST, ""),
    ],
)
def test_provider_attestation_rejects_missing_field(provider, digest, sig) -> None:
    """An attestation missing any part is not an attestation and is refused."""
    with pytest.raises(ValueError, match="must be non-empty"):
        build_provider_attestation(provider=provider, result_pack_digest=digest, provider_sig=sig)


# ── end-to-end sealing through the platform verdict ────────────────────


def test_sealed_pack_verifies_through_platform() -> None:
    """A sealed pack is attestation-mode and verifies VERIFIED with the ML-DSA key."""
    from scpn_studio_platform.seal import Keyring, Verdict, verify

    signer = _signer()
    unit = build_result_pack_unit(
        _real_pack(), raw_results_digest=_RAW_DIGEST, circuit_digest=_CIRCUIT_DIGEST
    )
    envelope = seal_result_pack(unit, signer=signer, attestation=_attestation(), grader=_GRADER)
    assert envelope.verifiability_mode == "attestation"
    assert envelope.attestation is not None
    assert envelope.attestation["provider"] == "ibm"
    assert envelope.signature["alg"] == "ML-DSA-65"

    ring = Keyring()
    ring.add(signer.key_id, signer.verifier())
    verdict = verify(envelope.to_dict(), None, keyring=ring, regrade=lambda _u: "bounded-support")
    assert verdict is Verdict.VERIFIED


def test_tampered_sealed_pack_is_forged() -> None:
    """Rewriting the sealed unit after attestation is caught as a forgery."""
    from scpn_studio_platform.seal import Keyring, Verdict, verify

    signer = _signer()
    unit = build_result_pack_unit(_real_pack(), raw_results_digest=_RAW_DIGEST)
    envelope = seal_result_pack(
        unit, signer=signer, attestation=_attestation(), grader=_GRADER
    ).to_dict()
    envelope["unit"] = {**unit, "claim_status": "reference-validated"}

    ring = Keyring()
    ring.add(signer.key_id, signer.verifier())
    verdict = verify(envelope, None, keyring=ring, regrade=lambda _u: "reference-validated")
    assert verdict is Verdict.FORGED


def test_seal_refuses_empty_attestation() -> None:
    """A QPU pack with no provider attestation is never sealed as verified."""
    signer = _signer()
    unit = build_result_pack_unit(_real_pack(), raw_results_digest=_RAW_DIGEST)
    with pytest.raises(ValueError, match="requires a provider attestation"):
        seal_result_pack(unit, signer=signer, attestation={}, grader=_GRADER)
