# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — QPU result-pack emission bridge tests
"""Tests for the provider-neutral ``studio.qpu-result-pack.v1`` emission bridge.

These lock the bridge that turns a live :class:`QuantumJobResult` — the neutral
result every one of the sixteen HAL adapters returns — into a WS-1
attestation-verifiable unit. The end-to-end path is exercised through the offline
:class:`LocalDeterministicSimulator`, an ungated real execution, and the counts
digest is pinned byte-for-byte against the executive deploy template so the
on-device provider signature and the studio-side digest can never drift.
"""

from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("scpn_studio_platform.seal", reason="studio extra not installed")

from scpn_quantum_control.hardware.hal import (
    BackendProfile,
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumJobRef,
    QuantumJobResult,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.qpu_result_pack_bridge import (
    job_result_provenance,
    qpu_result_pack_from_job,
    raw_results_digest,
)
from scpn_quantum_control.studio.qpu_result_pack import (
    QPU_VERIFIABILITY_MODE,
    present_qpu_result_pack,
)
from scpn_quantum_control.studio.verbs import QPU_RESULT_PACK_SCHEMA

_COUNTS: dict[str, int] = {"0000": 520, "1111": 504}
_NON_CLAIMS: tuple[str, ...] = ("broad quantum advantage", "multi-device replication")


def _local_profile(backend_id: str = "local_statevector") -> BackendProfile:
    """Return a built-in non-cloud profile for offline execution."""
    return HardwareAbstractionLayer.with_builtin_profiles().profile(backend_id)


def _run_local_job() -> tuple[QuantumJobResult, BackendProfile]:
    """Execute a real offline job and return its neutral result and profile."""
    profile = _local_profile()
    simulator = LocalDeterministicSimulator(profile)
    workload = QuantumWorkload(
        workload_id="ghz_4q",
        ir_format="openqasm3",
        program="OPENQASM 3.0; qubit[4] q;",
        n_qubits=4,
        shots=1024,
        metadata={"seed": "7"},
    )
    job = simulator.submit(workload)
    return simulator.result(job), profile


def _result_with_counts(
    counts: dict[str, int],
    *,
    backend_id: str = "local_statevector",
    metadata: dict[str, object] | None = None,
) -> QuantumJobResult:
    """Build a completed neutral result carrying ``counts``."""
    job = QuantumJobRef(
        job_id=f"{backend_id}:wl:abc123",
        backend_id=backend_id,
        workload_id="wl",
        status="completed",
    )
    return QuantumJobResult(
        job=job,
        status="completed",
        counts=counts,
        shots=sum(counts.values()),
        metadata=metadata or {},
    )


# --------------------------------------------------------------------------- #
# raw_results_digest
# --------------------------------------------------------------------------- #
def test_raw_results_digest_matches_the_executive_template_algorithm() -> None:
    """The bridge digest is byte-for-byte the algorithm a provider signs against."""
    payload = json.dumps(_COUNTS, sort_keys=True, separators=(",", ":"))
    expected = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert raw_results_digest(_COUNTS) == expected


def test_raw_results_digest_is_order_independent() -> None:
    """Canonical JSON sorts keys, so count insertion order cannot change it."""
    reordered = {"1111": 504, "0000": 520}
    assert raw_results_digest(reordered) == raw_results_digest(_COUNTS)


def test_raw_results_digest_rejects_empty_counts() -> None:
    """An empty histogram carries no observation and fails closed."""
    with pytest.raises(ValueError, match="counts must be non-empty"):
        raw_results_digest({})


# --------------------------------------------------------------------------- #
# job_result_provenance
# --------------------------------------------------------------------------- #
def test_provenance_defaults_are_drawn_from_the_neutral_result() -> None:
    """Ids default to the job/profile; the family falls back to the modality."""
    result = _result_with_counts(_COUNTS)
    provenance = job_result_provenance(
        result,
        profile=_local_profile(),
        title="offline GHZ observation",
        executed_utc="2026-07-11",
        claim_scope="Supports the offline 4-qubit GHZ parity observation.",
        non_claims=_NON_CLAIMS,
    )
    assert provenance["id"] == result.job.job_id
    assert provenance["backend"] == "local_statevector"
    # local_statevector carries no target_family, so the modality is used.
    assert provenance["hardware_family"] == "simulator"
    assert provenance["required_job_ids"] == [result.job.job_id]
    assert provenance["non_claims"] == list(_NON_CLAIMS)


def test_provenance_prefers_the_result_metadata_backend() -> None:
    """A device name in the result metadata names the concrete backend."""
    result = _result_with_counts(_COUNTS, metadata={"backend": "ibm_kingston"})
    provenance = job_result_provenance(
        result,
        profile=HardwareAbstractionLayer.with_builtin_profiles().profile("ibm_quantum"),
        title="live IBM observation",
        executed_utc="2026-07-11",
        claim_scope="Supports a bounded IBM Heron observation.",
        non_claims=_NON_CLAIMS,
    )
    assert provenance["backend"] == "ibm_kingston"
    # ibm_quantum carries an explicit target_family.
    assert provenance["hardware_family"] == "ibm_quantum"


def test_provenance_overrides_take_precedence() -> None:
    """Explicit id/backend/family overrides win over the derived defaults."""
    result = _result_with_counts(_COUNTS, metadata={"backend": "ibm_kingston"})
    provenance = job_result_provenance(
        result,
        profile=HardwareAbstractionLayer.with_builtin_profiles().profile("ibm_quantum"),
        title="live IBM observation",
        executed_utc="2026-07-11",
        claim_scope="Supports a bounded IBM Heron observation.",
        non_claims=_NON_CLAIMS,
        pack_id="phase1_ibm_2026_07",
        backend="ibm_marrakesh",
        hardware_family="IBM Heron r2",
    )
    assert provenance["id"] == "phase1_ibm_2026_07"
    assert provenance["backend"] == "ibm_marrakesh"
    assert provenance["hardware_family"] == "IBM Heron r2"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("title", "  ", "title must be non-empty"),
        ("executed_utc", "", "executed_utc must be non-empty"),
        ("claim_scope", " ", "claim_scope must be non-empty"),
    ],
)
def test_provenance_rejects_blank_scoping(field: str, value: str, match: str) -> None:
    """Claim scoping cannot be inferred from counts, so blanks fail closed."""
    scoping: dict[str, str] = {
        "title": "offline GHZ observation",
        "executed_utc": "2026-07-11",
        "claim_scope": "Supports the offline observation.",
    }
    scoping[field] = value
    with pytest.raises(ValueError, match=match):
        job_result_provenance(
            _result_with_counts(_COUNTS),
            profile=_local_profile(),
            title=scoping["title"],
            executed_utc=scoping["executed_utc"],
            claim_scope=scoping["claim_scope"],
            non_claims=_NON_CLAIMS,
        )


def test_provenance_requires_a_stated_non_claim() -> None:
    """An honest pack always states at least one boundary it does not claim."""
    with pytest.raises(ValueError, match="non_claims must state"):
        job_result_provenance(
            _result_with_counts(_COUNTS),
            profile=_local_profile(),
            title="offline GHZ observation",
            executed_utc="2026-07-11",
            claim_scope="Supports the offline observation.",
            non_claims=(),
        )


# --------------------------------------------------------------------------- #
# qpu_result_pack_from_job — end to end
# --------------------------------------------------------------------------- #
def test_offline_run_emits_an_unverifiable_unit_without_attestation() -> None:
    """A real offline run yields a well-formed unit that is honestly unverifiable."""
    result, profile = _run_local_job()
    unit = qpu_result_pack_from_job(
        result,
        profile=profile,
        title="offline GHZ parity observation",
        executed_utc="2026-07-11",
        claim_scope="Supports the offline 4-qubit GHZ parity observation.",
        non_claims=_NON_CLAIMS,
        circuit_digest="sha256:" + "22" * 32,
        calibration_ref="calibration/local/2026-07-11.json",
    )
    assert unit["schema"] == QPU_RESULT_PACK_SCHEMA
    assert unit["verifiability_mode"] == QPU_VERIFIABILITY_MODE == "attestation"
    assert unit["raw_results_digest"] == raw_results_digest(result.counts)
    assert unit["circuit_digest"] == "sha256:" + "22" * 32
    assert unit["provenance"]["required_job_ids"] == [result.job.job_id]
    assert "attestation" not in unit
    presentation = present_qpu_result_pack(unit)
    assert presentation.status == "unverifiable"


def test_attested_run_renders_and_seals_attestation_verifiable() -> None:
    """A provider attestation over the counts digest makes the unit verifiable."""
    pytest.importorskip("scpn_studio_platform.seal", reason="studio extra not installed")
    from scpn_quantum_control.crypto.ml_dsa_seal import MLDSASigner
    from scpn_quantum_control.studio.qpu_result_pack import seal_qpu_result_pack

    result, profile = _run_local_job()
    digest = raw_results_digest(result.counts)
    attestation = {
        "provider": "scpn-offline",
        "result_pack_digest": digest,
        "provider_sig": "cHJvdmlkZXItc2ln",
    }
    unit = qpu_result_pack_from_job(
        result,
        profile=profile,
        title="offline GHZ parity observation",
        executed_utc="2026-07-11",
        claim_scope="Supports the offline 4-qubit GHZ parity observation.",
        non_claims=_NON_CLAIMS,
        attestation=attestation,
    )
    presentation = present_qpu_result_pack(unit)
    assert presentation.status == "attestation-verifiable"
    signer = MLDSASigner.generate("scpn-quantum-control:qpu", seed=bytes(range(32)))
    envelope = seal_qpu_result_pack(
        unit, signer=signer, grader={"name": "honesty-bridge", "version": "0.8.0"}
    )
    assert envelope.verifiability_mode == QPU_VERIFIABILITY_MODE


def test_from_job_rejects_a_mismatched_attestation() -> None:
    """An attestation signing a different digest fails closed at emission."""
    result, profile = _run_local_job()
    with pytest.raises(ValueError, match="different digest"):
        qpu_result_pack_from_job(
            result,
            profile=profile,
            title="offline GHZ parity observation",
            executed_utc="2026-07-11",
            claim_scope="Supports the offline observation.",
            non_claims=_NON_CLAIMS,
            attestation={
                "provider": "scpn-offline",
                "result_pack_digest": "sha256:" + "99" * 32,
                "provider_sig": "cHJvdmlkZXItc2ln",
            },
        )


def test_from_job_rejects_a_result_with_no_counts() -> None:
    """A completed result with no counts cannot be digested into a pack."""
    empty = QuantumJobResult(
        job=QuantumJobRef(
            job_id="local_statevector:wl:def456",
            backend_id="local_statevector",
            workload_id="wl",
            status="completed",
        ),
        status="completed",
    )
    with pytest.raises(ValueError, match="counts must be non-empty"):
        qpu_result_pack_from_job(
            empty,
            profile=_local_profile(),
            title="offline observation",
            executed_utc="2026-07-11",
            claim_scope="Supports nothing measurable.",
            non_claims=_NON_CLAIMS,
        )
