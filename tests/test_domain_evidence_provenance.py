# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — domain and sensing evidence keeps its provenance
"""Synthetic stays synthetic, splits must match, tests never qualify entropy, data needs authority.

Oracles: a simulated NV ODMR scan whose true field is known, two splits built
from disjoint sample identifiers, bytes signed with an HMAC key held here, and
a fetch function that records every call it receives.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json

import numpy as np
import pytest

from scpn_quantum_control.sensing.domain_evidence import (
    DOMAIN_EVIDENCE_CLAIM_BOUNDARY,
    ENTROPY_CLAIM_BOUNDARY,
    DomainEvidence,
    EntropyEvidence,
    ExternalDatasetRequest,
    acquire_external_dataset,
    assess_random_bytes,
    compare_evidence,
    nv_field_evidence,
    split_digest,
)
from scpn_quantum_control.sensing.nv_magnetometry_20T import (
    NVCenter,
    calibrate_field_from_odmr,
    odmr_resonances_hz,
    simulate_odmr_measurement,
)

_FIELD_T = 0.5
_SPLIT_A = tuple(f"scan-{index:03d}" for index in range(8))
_SPLIT_B = tuple(f"scan-{index:03d}" for index in range(8, 16))


def _nv_evidence(sample_ids: tuple[str, ...]) -> DomainEvidence:
    """Calibrate a simulated 0.5 T ODMR scan and wrap it as evidence on the given split."""
    nv = NVCenter()
    upper = odmr_resonances_hz(nv, _FIELD_T)[1]
    freqs = np.linspace(upper - 5.0e7, upper + 5.0e7, 4000)
    measured = simulate_odmr_measurement(freqs, nv, _FIELD_T, noise_std=0.004, seed=5)
    calibration = calibrate_field_from_odmr(freqs, measured, nv, true_field_tesla=_FIELD_T)
    return nv_field_evidence(calibration, sample_ids=sample_ids, null_baseline_tesla=_FIELD_T)


def test_domain_evidence_provenance_01_synthetic_label_survives_export() -> None:
    """A simulated NV result exports and re-imports as synthetic with its provenance intact."""
    evidence = _nv_evidence(_SPLIT_A)
    assert evidence.kind == "synthetic" and evidence.use_authority is None
    assert evidence.value < 5.0e-5  # the calibration recovers the simulated field
    assert evidence.improvement_over_null == pytest.approx(evidence.value - _FIELD_T)
    exported = json.loads(evidence.to_json())
    assert exported["kind"] == "synthetic"
    assert exported["split"] == split_digest(_SPLIT_A)
    assert exported["claim_boundary"] == DOMAIN_EVIDENCE_CLAIM_BOUNDARY
    restored = DomainEvidence.from_json(evidence.to_json())
    assert restored == evidence and restored.kind == "synthetic"
    # Relabelling the export as empirical data is refused: no use authority exists for it.
    exported["kind"] = "empirical"
    with pytest.raises(ValueError, match="empirical evidence needs a recorded use authority"):
        DomainEvidence.from_json(json.dumps(exported))
    # A synthetic result compared with a proxy result of the same numbers is refused.
    proxy = dataclasses.replace(evidence, kind="proxy")
    with pytest.raises(ValueError, match="differs in kind"):
        compare_evidence(evidence, proxy)


def test_domain_evidence_provenance_02_same_numbers_on_another_split_are_not_compared() -> None:
    """Identical values on different splits refuse comparison; the same split compares."""
    first = _nv_evidence(_SPLIT_A)
    other_split = _nv_evidence(_SPLIT_B)
    assert first.value == other_split.value and first.split != other_split.split
    with pytest.raises(ValueError, match="differs in split"):
        compare_evidence(first, other_split)
    # The split identity does not depend on order, so a reshuffled list is the same split.
    reshuffled = _nv_evidence(tuple(reversed(_SPLIT_A)))
    assert compare_evidence(first, reshuffled) == 0.0
    renamed = dataclasses.replace(first, estimand="relative axial-field error")
    with pytest.raises(ValueError, match="differs in estimand"):
        compare_evidence(first, renamed)


def test_domain_evidence_provenance_03_signed_random_bytes_do_not_qualify_entropy() -> None:
    """Passing tests and a valid signature leave source entropy unqualified."""
    key = b"held-here-for-the-test"
    payload = np.random.default_rng(7).integers(0, 256, size=4096, dtype=np.uint8).tobytes()
    tag = hmac.new(key, payload, hashlib.sha256).digest()

    def verify(data: bytes) -> bool:
        """Check the HMAC tag of the bytes."""
        return hmac.compare_digest(hmac.new(key, data, hashlib.sha256).digest(), tag)

    assessed = assess_random_bytes(payload, verify_signature=verify)
    assert assessed.frequency_passed and assessed.runs_passed and assessed.signature_valid
    assert assessed.source_entropy_qualified is False
    assert assessed.claim_boundary == ENTROPY_CLAIM_BOUNDARY
    # A fully predictable, correctly signed payload is reported the same way: tests fail, signature holds.
    constant = bytes(4096)
    constant_tag = hmac.new(key, constant, hashlib.sha256).digest()
    predictable = assess_random_bytes(
        constant,
        verify_signature=lambda data: hmac.compare_digest(
            hmac.new(key, data, hashlib.sha256).digest(), constant_tag
        ),
    )
    assert not predictable.frequency_passed and predictable.signature_valid
    assert predictable.source_entropy_qualified is False
    tampered = assess_random_bytes(payload[:-1] + b"\x00", verify_signature=verify)
    assert not tampered.signature_valid
    with pytest.raises(ValueError, match="cannot qualify source entropy"):
        EntropyEvidence(True, True, True, source_entropy_qualified=True)


def test_domain_evidence_provenance_04_external_data_without_authority_is_not_acquired() -> None:
    """No use authority means no fetch call; a recorded authority fetches once."""
    calls: list[str] = []

    def fetch(dataset_id: str) -> bytes:
        """Record the request and return placeholder bytes."""
        calls.append(dataset_id)
        return b"rows"

    for authority in (None, ""):
        with pytest.raises(PermissionError, match="no recorded use authority"):
            acquire_external_dataset(
                ExternalDatasetRequest("clinical-eeg-cohort", "clinical", authority), fetch
            )
    assert calls == []
    with pytest.raises(ValueError, match="generated here, not acquired"):
        acquire_external_dataset(
            ExternalDatasetRequest("simulated-scan", "synthetic", "not needed"), fetch
        )
    assert calls == []
    licensed = ExternalDatasetRequest("public-magnetometer-log", "empirical", "CC-BY-4.0")
    assert acquire_external_dataset(licensed, fetch) == b"rows"
    assert calls == ["public-magnetometer-log"]


def test_evidence_validation_refusals() -> None:
    """Missing provenance, unknown kinds, non-finite values and foreign schemas are refused."""
    base = _nv_evidence(_SPLIT_A)
    with pytest.raises(ValueError, match="estimand is required"):
        dataclasses.replace(base, estimand="  ")
    with pytest.raises(ValueError, match="source is required"):
        dataclasses.replace(base, source="")
    with pytest.raises(ValueError, match="split is required"):
        dataclasses.replace(base, split=" ")
    with pytest.raises(ValueError, match="evidence kind"):
        DomainEvidence.from_json(base.to_json().replace('"synthetic"', '"anecdotal"'))
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(base, value=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(base, null_baseline=float("inf"))
    with pytest.raises(ValueError, match="schema"):
        dataclasses.replace(base, schema="domain_evidence.v0")
    with pytest.raises(ValueError, match="claim boundary"):
        dataclasses.replace(base, claim_boundary="efficacy demonstrated")
    clinical = dataclasses.replace(base, kind="clinical", use_authority="ethics approval 2026-17")
    assert clinical.kind == "clinical"
    with pytest.raises(ValueError, match="at least one sample"):
        split_digest(())
    with pytest.raises(ValueError, match="unique"):
        split_digest(("a", "a"))
    nv = NVCenter()
    upper = odmr_resonances_hz(nv, _FIELD_T)[1]
    freqs = np.linspace(upper - 5.0e7, upper + 5.0e7, 400)
    blind = calibrate_field_from_odmr(freqs, simulate_odmr_measurement(freqs, nv, _FIELD_T), nv)
    with pytest.raises(ValueError, match="true field"):
        nv_field_evidence(blind, sample_ids=_SPLIT_A, null_baseline_tesla=_FIELD_T)
    with pytest.raises(ValueError, match="at least 16 bytes"):
        assess_random_bytes(b"short", verify_signature=lambda data: True)
