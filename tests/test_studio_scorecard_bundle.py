# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio baseline-scorecard bundle tests
"""Tests for the schema-B baseline-scorecard bundle emitter (ST-02)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_studio_platform.evidence import EvidenceBundle  # noqa: E402

from scpn_quantum_control.differentiable_baseline_scorecard import (  # noqa: E402
    REQUIRED_BASELINE_CATEGORIES,
)
from scpn_quantum_control.differentiable_baseline_scorecard import (  # noqa: E402
    run_differentiable_baseline_scorecard as run_baseline_scorecard,
)
from scpn_quantum_control.studio import scorecard_bundle  # noqa: E402
from scpn_quantum_control.studio.evidence_bundle import (  # noqa: E402
    StudioBundleValidation,
    validate_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundle_is_admitted_by_the_federation_gate() -> None:
    """The emitted bundle passes the platform schema-B validation gate."""
    validated = validate_bundle(scorecard_bundle.build_scorecard_bundle())
    assert validated.verdict.admitted, validated.verdict.rejections
    assert validated.bundle.schema == "studio.differentiation-evidence.v1"


def test_bundle_carries_all_categories_with_verbatim_statuses() -> None:
    """All eleven category rows ride in cases[] with untouched statuses."""
    scorecard = run_baseline_scorecard()
    bundle = scorecard_bundle.build_scorecard_bundle(scorecard)
    families = [case.operation_family for case in bundle.cases]
    assert families == [
        f"baseline-category:{category}" for category in REQUIRED_BASELINE_CATEGORIES
    ]
    statuses = {case.status for case in bundle.cases}
    assert statuses == {row.status for row in scorecard.rows}
    assert [case.dimension for case in bundle.cases] == list(range(1, 12))


def test_bundle_never_upgrades_the_scorecard_boundary() -> None:
    """The bundle is bounded-model with the scorecard's own boundary note."""
    scorecard = run_baseline_scorecard()
    bundle = scorecard_bundle.build_scorecard_bundle(scorecard)
    boundary = bundle.claim_boundary
    assert boundary.status.value == "bounded-model"
    assert boundary.admission.value == "admitted"
    assert boundary.validity_domain is not None
    assert boundary.validity_domain.note == scorecard.claim_boundary
    wire = json.dumps(bundle.to_dict(), sort_keys=True)
    assert "reference-validated" not in wire
    assert bundle.activity.verb == "differentiate"


def test_bundle_digest_is_deterministic() -> None:
    """Two emissions of the committed scorecard share one entity digest."""
    first = scorecard_bundle.build_scorecard_bundle()
    second = scorecard_bundle.build_scorecard_bundle()
    assert first.entity.digest == second.entity.digest
    assert first.entity.digest.startswith("sha256:")


def test_artifact_edge_content_addresses_the_committed_artefact() -> None:
    """The optional derivation edge digests the committed artefact bytes."""
    artifact = REPO_ROOT / scorecard_bundle.DEFAULT_SCORECARD_ARTIFACT_PATH
    bundle = scorecard_bundle.build_scorecard_bundle(artifact_path=artifact)
    assert len(bundle.derived_from) == 1
    edge = bundle.derived_from[0]
    assert edge.entity_digest.startswith("sha256:")
    assert edge.studio == "scpn-quantum-control"
    without = scorecard_bundle.build_scorecard_bundle()
    assert without.derived_from == ()


def test_missing_artifact_path_fails_closed(tmp_path: Path) -> None:
    """A requested derivation edge with no artefact refuses to emit."""
    with pytest.raises(ValueError, match="scorecard artefact does not exist"):
        scorecard_bundle.build_scorecard_bundle(artifact_path=tmp_path / "missing.json")


def test_invalid_scorecard_is_never_federated() -> None:
    """A scorecard failing its own validation raises instead of emitting."""
    scorecard = run_baseline_scorecard()
    tampered = dataclasses.replace(scorecard, artifact_id="forged-artifact")
    with pytest.raises(ValueError, match="scorecard failed validation"):
        scorecard_bundle.build_scorecard_bundle(tampered)


def test_main_emits_admitted_bundle_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints the wire bundle and exits 0 on admission."""
    artifact = REPO_ROOT / scorecard_bundle.DEFAULT_SCORECARD_ARTIFACT_PATH
    exit_code = scorecard_bundle.main(["--artifact-path", str(artifact)])
    captured = capsys.readouterr()
    assert exit_code == 0
    wire = json.loads(captured.out)
    assert wire["schema"] == "studio.differentiation-evidence.v1"
    assert len(wire["cases"]) == 11
    assert captured.err == ""


def test_main_fails_closed_on_missing_artifact(tmp_path: Path) -> None:
    """The CLI propagates the fail-closed artefact error."""
    with pytest.raises(ValueError, match="scorecard artefact does not exist"):
        scorecard_bundle.main(["--artifact-path", str(tmp_path / "missing.json")])


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
            rejections=("forced rejection for the CLI branch",),
        )
        return dataclasses.replace(validated, verdict=verdict)

    monkeypatch.setattr(scorecard_bundle, "validate_bundle", rejecting_validate)
    artifact = REPO_ROOT / scorecard_bundle.DEFAULT_SCORECARD_ARTIFACT_PATH
    exit_code = scorecard_bundle.main(["--artifact-path", str(artifact)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "forced rejection for the CLI branch" in captured.err
