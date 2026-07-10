# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio QEC-readiness bundle tests
"""Tests for the schema-B ``studio.qec-readiness.v1`` bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform.evidence", reason="studio extra not installed")

from scpn_quantum_control.differentiable_claim_ledger import REPO_ROOT  # noqa: E402
from scpn_quantum_control.studio import qec_readiness_bundle, verbs  # noqa: E402
from scpn_quantum_control.studio.evidence_bundle import validate_bundle  # noqa: E402

COMMITTED_ARTIFACT = REPO_ROOT / qec_readiness_bundle.DEFAULT_QEC_READINESS_ARTIFACT_PATH


def _committed_payload() -> dict:
    return json.loads(COMMITTED_ARTIFACT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# verb-contract wiring
# --------------------------------------------------------------------------- #
def test_qec_readiness_schema_is_produced_by_the_validate_verb() -> None:
    assert verbs.QEC_READINESS_SCHEMA == "studio.qec-readiness.v1"
    assert verbs.QEC_READINESS_SCHEMA in verbs.VALIDATE.produces
    assert verbs.QEC_READINESS_SCHEMA in verbs.evidence_schemas()


# --------------------------------------------------------------------------- #
# committed-artefact federation
# --------------------------------------------------------------------------- #
def test_bundle_federates_every_decoder_aggregate_verbatim() -> None:
    payload = _committed_payload()
    bundle = qec_readiness_bundle.build_qec_readiness_bundle()
    assert bundle.schema == "studio.qec-readiness.v1"
    assert len(bundle.cases) == len(payload["decoder_aggregates"])
    for case, aggregate in zip(bundle.cases, payload["decoder_aggregates"], strict=True):
        assert case.operation_family == (
            f"qec-decoder:{aggregate['family']}:{aggregate['label']}"
            f":{aggregate['decoder']}:{aggregate['noise_model']}"
        )
        assert case.dimension == int(payload["distance"])
        assert case.status == "simulated"
        assert case.error == pytest.approx(float(aggregate["logical_failure_rate"]))


def test_bundle_claim_boundary_carries_supported_and_blocked_verbatim() -> None:
    payload = _committed_payload()
    bundle = qec_readiness_bundle.build_qec_readiness_bundle()
    note = bundle.claim_boundary.validity_domain.note
    assert payload["readiness_decision"] in note
    for blocked in payload["claim_boundary"]["blocked"]:
        assert blocked in note
    for supported in payload["claim_boundary"]["supported"]:
        assert supported in note
    assert bundle.claim_boundary.status.value == "bounded-model"


def test_bundle_provenance_axes_are_offline_numerical() -> None:
    bundle = qec_readiness_bundle.build_qec_readiness_bundle()
    assert bundle.substrate.value == "numerical-model"
    assert bundle.evidence_kind.value == "measured"
    assert bundle.numeric_provenance.active_backend == "offline-surface-code-decoders"
    assert bundle.numeric_provenance.reference_backend == "unencoded-physical"


def test_bundle_derivation_edge_digests_the_committed_artifact() -> None:
    bundle = qec_readiness_bundle.build_qec_readiness_bundle()
    (edge,) = bundle.derived_from
    assert edge.entity_digest.startswith("sha256:")


def test_bundle_is_admitted_by_the_federation_gate() -> None:
    validated = validate_bundle(qec_readiness_bundle.build_qec_readiness_bundle())
    assert validated.verdict.admitted, validated.verdict.rejections


# --------------------------------------------------------------------------- #
# fail-closed shape checks
# --------------------------------------------------------------------------- #
def _drifted_artifact(tmp_path: Path, mutate: dict) -> Path:
    payload = _committed_payload()
    payload.update(mutate)
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_empty_aggregates_fail_closed(tmp_path: Path) -> None:
    path = _drifted_artifact(tmp_path, {"decoder_aggregates": []})
    with pytest.raises(ValueError, match="no decoder aggregates"):
        qec_readiness_bundle.build_qec_readiness_bundle(artifact_path=path)


@pytest.mark.parametrize("bad_distance", [0, True, "three", None])
def test_bad_distance_fails_closed(tmp_path: Path, bad_distance: object) -> None:
    path = _drifted_artifact(tmp_path, {"distance": bad_distance})
    with pytest.raises(ValueError, match="code distance"):
        qec_readiness_bundle.build_qec_readiness_bundle(artifact_path=path)


def test_missing_decision_fails_closed(tmp_path: Path) -> None:
    path = _drifted_artifact(tmp_path, {"readiness_decision": ""})
    with pytest.raises(ValueError, match="readiness decision"):
        qec_readiness_bundle.build_qec_readiness_bundle(artifact_path=path)


@pytest.mark.parametrize(
    "boundary",
    [
        None,
        {"blocked": [], "supported": ["x"]},
        {"blocked": ["x"], "supported": []},
        {"supported": ["x"]},
    ],
)
def test_missing_claim_boundary_fails_closed(tmp_path: Path, boundary: object) -> None:
    path = _drifted_artifact(tmp_path, {"claim_boundary": boundary})
    with pytest.raises(ValueError, match="supported/blocked claim boundary"):
        qec_readiness_bundle.build_qec_readiness_bundle(artifact_path=path)


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        qec_readiness_bundle.build_qec_readiness_bundle(artifact_path=tmp_path / "absent.json")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_main_emits_admitted_bundle_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = qec_readiness_bundle.main(["--artifact-path", str(COMMITTED_ARTIFACT)])
    captured = capsys.readouterr()
    assert exit_code == 0
    wire = json.loads(captured.out)
    assert wire["schema"] == "studio.qec-readiness.v1"
    assert len(wire["cases"]) == len(_committed_payload()["decoder_aggregates"])
    assert all(
        isinstance(case["error"], str) and case["status"] == "simulated" for case in wire["cases"]
    )


def test_main_default_artifact_path(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = qec_readiness_bundle.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["schema"] == "studio.qec-readiness.v1"


def test_main_returns_one_when_the_gate_rejects(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RejectedVerdict:
        admitted = False
        rejections = ("forced rejection",)

    class _RejectedValidation:
        verdict = _RejectedVerdict()
        bundle = qec_readiness_bundle.build_qec_readiness_bundle()

    monkeypatch.setattr(
        qec_readiness_bundle, "validate_bundle", lambda bundle: _RejectedValidation()
    )
    exit_code = qec_readiness_bundle.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "forced rejection" in captured.err
