# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio coupling-invariant bundle tests
"""Tests for the schema-B effective-coupling invariant bundle emitter (ST-14)."""

from __future__ import annotations

import dataclasses
import json

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_studio_platform.evidence import EvidenceBundle  # noqa: E402

from scpn_quantum_control.studio import coupling_invariant  # noqa: E402
from scpn_quantum_control.studio.evidence_bundle import (  # noqa: E402
    StudioBundleValidation,
    validate_bundle,
)
from scpn_quantum_control.studio.verbs import (  # noqa: E402
    ANALYSE,
    COUPLING_INVARIANT_SCHEMA,
    DLA_PARITY_SCHEMA,
)


def test_payload_names_exact_effective_coupling_invariant() -> None:
    """The payload is the ST-14 `knm.kuramoto.effective-coupling` invariant."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    assert coupling_invariant.validate_coupling_invariant_payload(payload)
    assert payload.schema == "studio.coupling-invariant.v1"
    assert payload.invariant_id == "knm.kuramoto.effective-coupling"
    assert "dla" not in payload.invariant_id.lower()


def test_payload_requires_the_two_committed_estimator_classes() -> None:
    """The invariant is sourced from Hamiltonian learning and parameter-shift coupling learning."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    by_id = {source.source_id: source for source in payload.estimator_sources}
    assert set(by_id) == {
        "hamiltonian-learning",
        "differentiable-coupling-learning.parameter-shift",
    }
    assert by_id["hamiltonian-learning"].class_name == "HamiltonianLearningResult"
    assert (
        by_id["differentiable-coupling-learning.parameter-shift"].class_name
        == "CouplingGradientVerificationResult"
    )
    assert all(source.role == "estimator" for source in by_id.values())


def test_payload_requires_sync_and_zne_uncertainty_sources() -> None:
    """Both mandated UQ sources are present and separately labelled."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    by_id = {source.source_id: source for source in payload.uncertainty_sources}
    assert set(by_id) == {"sync_uncertainty", "zne_uncertainty"}
    assert by_id["sync_uncertainty"].class_name == "UncertaintyInterval"
    assert by_id["zne_uncertainty"].class_name == "ZNEUncertaintyResult"
    assert all(source.role == "uncertainty" for source in by_id.values())


def test_bundle_is_admitted_by_the_federation_gate() -> None:
    """The emitted coupling-invariant bundle passes the platform schema-B gate."""
    validated = validate_bundle(coupling_invariant.build_coupling_invariant_bundle())
    assert validated.verdict.admitted, validated.verdict.rejections
    assert validated.bundle.schema == COUPLING_INVARIANT_SCHEMA
    assert validated.bundle.activity.verb == ANALYSE.name
    assert validated.bundle.claim_boundary.status.value == "bounded-model"


def test_bundle_cases_cover_estimators_then_uncertainty_sources() -> None:
    """All four sources ride in cases[] under the effective-coupling operation family."""
    bundle = coupling_invariant.build_coupling_invariant_bundle()
    families = [case.operation_family for case in bundle.cases]
    assert families == [
        "knm.kuramoto.effective-coupling:hamiltonian-learning",
        "knm.kuramoto.effective-coupling:differentiable-coupling-learning.parameter-shift",
        "knm.kuramoto.effective-coupling:sync_uncertainty",
        "knm.kuramoto.effective-coupling:zne_uncertainty",
    ]
    assert [case.dimension for case in bundle.cases] == [1, 2, 3, 4]
    assert [case.status for case in bundle.cases] == [
        "estimator",
        "estimator",
        "uncertainty",
        "uncertainty",
    ]


def test_coupling_invariant_schema_is_not_the_dla_schema() -> None:
    """ST-14 is an analyse schema but remains separate from DLA parity."""
    assert COUPLING_INVARIANT_SCHEMA != DLA_PARITY_SCHEMA
    assert COUPLING_INVARIANT_SCHEMA in ANALYSE.produces
    assert DLA_PARITY_SCHEMA in ANALYSE.produces


def test_bundle_digest_is_deterministic() -> None:
    """Two emissions over the static source inventory share one digest."""
    first = coupling_invariant.build_coupling_invariant_bundle()
    second = coupling_invariant.build_coupling_invariant_bundle()
    assert first.entity.digest == second.entity.digest
    assert first.entity.digest.startswith("sha256:")


def test_payload_validation_fails_closed_when_estimator_is_missing() -> None:
    """Dropping either estimator source refuses to emit the invariant."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    tampered = dataclasses.replace(payload, estimator_sources=payload.estimator_sources[:1])
    with pytest.raises(ValueError, match="two committed estimator"):
        coupling_invariant.validate_coupling_invariant_payload(tampered)


def test_payload_validation_fails_closed_when_uncertainty_is_missing() -> None:
    """Dropping either UQ source refuses to emit the invariant."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    tampered = dataclasses.replace(payload, uncertainty_sources=payload.uncertainty_sources[:1])
    with pytest.raises(ValueError, match="sync_uncertainty and zne_uncertainty"):
        coupling_invariant.validate_coupling_invariant_payload(tampered)


def test_payload_validation_fails_closed_for_wrong_invariant_id() -> None:
    """An invariant id drift cannot be federated."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    tampered = dataclasses.replace(payload, invariant_id="knm.kuramoto.other")
    with pytest.raises(ValueError, match="effective-coupling"):
        coupling_invariant.validate_coupling_invariant_payload(tampered)


def test_payload_validation_fails_closed_for_dla_invariant_id() -> None:
    """A DLA invariant id is refused before exact-id validation."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    tampered = dataclasses.replace(payload, invariant_id="knm.kuramoto.dla")
    with pytest.raises(ValueError, match="DLA invariant"):
        coupling_invariant.validate_coupling_invariant_payload(tampered)


def test_payload_validation_fails_closed_for_wrong_schema() -> None:
    """The Studio schema name is load-bearing."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    tampered = dataclasses.replace(payload, schema="studio.other.v1")
    with pytest.raises(ValueError, match="wrong schema"):
        coupling_invariant.validate_coupling_invariant_payload(tampered)


def test_payload_validation_fails_closed_for_wrong_source_roles() -> None:
    """Estimator and UQ source roles are checked separately."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    bad_estimator = dataclasses.replace(payload.estimator_sources[0], role="uncertainty")
    bad_uncertainty = dataclasses.replace(payload.uncertainty_sources[0], role="estimator")
    with pytest.raises(ValueError, match="role=estimator"):
        coupling_invariant.validate_coupling_invariant_payload(
            dataclasses.replace(
                payload,
                estimator_sources=(bad_estimator, payload.estimator_sources[1]),
            )
        )
    with pytest.raises(ValueError, match="role=uncertainty"):
        coupling_invariant.validate_coupling_invariant_payload(
            dataclasses.replace(
                payload,
                uncertainty_sources=(bad_uncertainty, payload.uncertainty_sources[1]),
            )
        )


def test_payload_validation_fails_closed_without_claim_boundary() -> None:
    """The bundle must carry a claim boundary before it can be emitted."""
    payload = coupling_invariant.build_coupling_invariant_payload()
    tampered = dataclasses.replace(payload, claim_boundary="")
    with pytest.raises(ValueError, match="claim boundary"):
        coupling_invariant.validate_coupling_invariant_payload(tampered)


def test_main_emits_admitted_bundle_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints the wire bundle and exits 0 on admission."""
    exit_code = coupling_invariant.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    wire = json.loads(captured.out)
    assert wire["schema"] == "studio.coupling-invariant.v1"
    assert len(wire["cases"]) == 4
    assert captured.err == ""


def test_main_reports_a_rejected_bundle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A federation-gate rejection exits 1 and names the rejections."""

    def rejecting_validate(bundle: EvidenceBundle) -> StudioBundleValidation:
        validated = validate_bundle(bundle)
        verdict = dataclasses.replace(
            validated.verdict,
            admitted=False,
            rejections=("forced coupling-invariant rejection",),
        )
        return dataclasses.replace(validated, verdict=verdict)

    monkeypatch.setattr(coupling_invariant, "validate_bundle", rejecting_validate)
    exit_code = coupling_invariant.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "forced coupling-invariant rejection" in captured.err
