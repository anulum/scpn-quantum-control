# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — WS-1 QPU result-pack unit tests
"""Tests for the attestation-verifiable ``studio.qpu-result-pack.v1`` unit (ST-10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("scpn_studio_platform.seal", reason="studio extra not installed")

from scpn_quantum_control.crypto.ml_dsa_seal import MLDSASigner  # noqa: E402
from scpn_quantum_control.studio.qpu_result_pack import (  # noqa: E402
    QPU_VERIFIABILITY_MODE,
    QpuResultPackPresentation,
    build_qpu_result_pack_unit,
    present_qpu_result_pack,
    seal_qpu_result_pack,
)
from scpn_quantum_control.studio.verbs import QPU_RESULT_PACK_SCHEMA  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO_ROOT / "data" / "hardware_result_packs" / "manifest.json"
_SEED = bytes(range(32))
_RAW_DIGEST = "sha256:" + "11" * 32
_CIRCUIT_DIGEST = "sha256:" + "22" * 32
_CALIBRATION_REF = "calibration/ibm_kingston/2026-04-10.json"
_GRADER = {"name": "honesty-bridge", "version": "0.8.0"}


def _real_pack() -> dict[str, Any]:
    packs: list[dict[str, Any]] = json.loads(_MANIFEST.read_text(encoding="utf-8"))["packs"]
    return packs[0]


def _signer() -> MLDSASigner:
    return MLDSASigner.generate("scpn-quantum-control:qpu", seed=_SEED)


def _attestation(digest: str = _RAW_DIGEST) -> dict[str, str]:
    return {
        "provider": "ibm",
        "result_pack_digest": digest,
        "provider_sig": "cHJvdmlkZXItc2ln",
    }


def test_unit_declares_the_attestation_verifiability_mode() -> None:
    """Every unit carries the WS-1 attestation mode and pack provenance."""
    unit = build_qpu_result_pack_unit(
        _real_pack(),
        raw_results_digest=_RAW_DIGEST,
        circuit_digest=_CIRCUIT_DIGEST,
        calibration_ref=_CALIBRATION_REF,
    )
    assert unit["schema"] == QPU_RESULT_PACK_SCHEMA
    assert unit["verifiability_mode"] == QPU_VERIFIABILITY_MODE == "attestation"
    assert unit["raw_results_digest"] == _RAW_DIGEST
    assert unit["circuit_digest"] == _CIRCUIT_DIGEST
    assert unit["calibration_ref"] == _CALIBRATION_REF
    assert unit["provenance"]["id"] == _real_pack()["id"]
    assert "attestation" not in unit


def test_optional_fields_are_omitted_when_absent() -> None:
    """A minimal unit omits the optional digest, calibration, and attestation."""
    unit = build_qpu_result_pack_unit(_real_pack(), raw_results_digest=_RAW_DIGEST)
    assert "circuit_digest" not in unit
    assert "calibration_ref" not in unit
    assert "attestation" not in unit


def test_unit_construction_rejects_empty_pack_id_and_digest() -> None:
    """A pack with no id or an empty digest fails closed at construction."""
    with pytest.raises(ValueError, match="non-empty id"):
        build_qpu_result_pack_unit({}, raw_results_digest=_RAW_DIGEST)
    with pytest.raises(ValueError, match="raw_results_digest must be"):
        build_qpu_result_pack_unit(_real_pack(), raw_results_digest="  ")


def test_supplied_attestation_must_sign_the_raw_results() -> None:
    """An attestation over a different digest is rejected at construction."""
    with pytest.raises(ValueError, match="different digest"):
        build_qpu_result_pack_unit(
            _real_pack(),
            raw_results_digest=_RAW_DIGEST,
            attestation=_attestation(digest="sha256:" + "99" * 32),
        )


def test_supplied_attestation_fields_must_be_present() -> None:
    """A missing attestation field fails closed at construction."""
    broken = _attestation()
    broken["provider_sig"] = ""
    with pytest.raises(ValueError, match="provider_sig"):
        build_qpu_result_pack_unit(
            _real_pack(), raw_results_digest=_RAW_DIGEST, attestation=broken
        )


def test_a_well_formed_attestation_binds_into_the_unit() -> None:
    """A valid attestation is normalised into the unit's attestation field."""
    unit = build_qpu_result_pack_unit(
        _real_pack(), raw_results_digest=_RAW_DIGEST, attestation=_attestation()
    )
    assert unit["attestation"] == _attestation()


def test_present_renders_unverifiable_without_an_attestation() -> None:
    """A committed pack with no live attestation renders unverifiable, loud."""
    unit = build_qpu_result_pack_unit(_real_pack(), raw_results_digest=_RAW_DIGEST)
    presentation = present_qpu_result_pack(unit)
    assert presentation.status == "unverifiable"
    assert "no provider attestation" in presentation.reason


def test_present_renders_attestation_verifiable_with_a_provider_signature() -> None:
    """A unit with a well-formed provider attestation is attestation-verifiable."""
    unit = build_qpu_result_pack_unit(
        _real_pack(), raw_results_digest=_RAW_DIGEST, attestation=_attestation()
    )
    presentation = present_qpu_result_pack(unit)
    assert presentation.status == "attestation-verifiable"
    assert "ibm" in presentation.reason


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        ({"schema": "studio.other.v1"}, "unknown qpu result-pack schema"),
        ({"verifiability_mode": "recompute"}, "not attestation"),
        ({"raw_results_digest": "  "}, "no raw-results digest"),
    ],
)
def test_present_strip_detection(mutation: dict[str, str], reason_fragment: str) -> None:
    """Stripping the schema, mode, or digest all render unverifiable."""
    unit = build_qpu_result_pack_unit(
        _real_pack(), raw_results_digest=_RAW_DIGEST, attestation=_attestation()
    )
    unit.update(mutation)
    presentation = present_qpu_result_pack(unit)
    assert presentation.status == "unverifiable"
    assert reason_fragment in presentation.reason


def test_present_detects_a_tampered_attestation_shape() -> None:
    """A hollowed-out or wrong-digest attestation renders unverifiable."""
    unit = build_qpu_result_pack_unit(
        _real_pack(), raw_results_digest=_RAW_DIGEST, attestation=_attestation()
    )
    hollow = dict(unit)
    hollow["attestation"] = {"provider": "ibm", "result_pack_digest": _RAW_DIGEST}
    assert present_qpu_result_pack(hollow).status == "unverifiable"
    assert "missing" in present_qpu_result_pack(hollow).reason
    mismatched = dict(unit)
    mismatched["attestation"] = _attestation(digest="sha256:" + "00" * 32)
    assert present_qpu_result_pack(mismatched).status == "unverifiable"
    empty = dict(unit)
    empty["attestation"] = {}
    assert present_qpu_result_pack(empty).status == "unverifiable"


def test_presentation_dataclass_invariants() -> None:
    """The presentation dataclass rejects unknown status or empty reason."""
    with pytest.raises(ValueError, match="status is unknown"):
        QpuResultPackPresentation("verified", "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason must be non-empty"):
        QpuResultPackPresentation("unverifiable", "  ")


def test_seal_refuses_an_unverifiable_unit() -> None:
    """A unit with no provider attestation is never sealed as verified."""
    unit = build_qpu_result_pack_unit(_real_pack(), raw_results_digest=_RAW_DIGEST)
    with pytest.raises(ValueError, match="never sealed as verified"):
        seal_qpu_result_pack(unit, signer=_signer(), grader=_GRADER)


def test_seal_produces_an_attestation_verifiable_envelope() -> None:
    """A well-attested unit seals and verifies through the platform keyring."""
    from scpn_studio_platform.seal import Keyring, Verdict, verify

    signer = _signer()
    unit = build_qpu_result_pack_unit(
        _real_pack(),
        raw_results_digest=_RAW_DIGEST,
        circuit_digest=_CIRCUIT_DIGEST,
        attestation=_attestation(),
    )
    envelope = seal_qpu_result_pack(unit, signer=signer, grader=_GRADER)
    assert envelope.attestation["provider"] == "ibm"
    ring = Keyring()
    ring.add(signer.key_id, signer.verifier())
    verdict = verify(envelope.to_dict(), None, keyring=ring, regrade=lambda _u: "bounded-support")
    assert verdict is Verdict.VERIFIED
