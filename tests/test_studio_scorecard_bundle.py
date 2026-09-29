# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio baseline-scorecard bundle tests
"""Tests for the schema-B baseline-scorecard bundle emitter."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from _qualification_receipt_vectors import ROOT as VECTOR_ROOT  # noqa: E402
from _qualification_receipt_vectors import build_receipt, seal  # noqa: E402
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


def test_module_entrypoint_emits_an_admitted_bundle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Execute the public module entrypoint and propagate its zero exit code."""
    module_name = "scpn_quantum_control.studio.scorecard_bundle"
    imported_module = sys.modules.pop(module_name)
    artifact = REPO_ROOT / scorecard_bundle.DEFAULT_SCORECARD_ARTIFACT_PATH
    monkeypatch.setattr(sys, "argv", [module_name, "--artifact-path", str(artifact)])
    try:
        with pytest.raises(SystemExit) as raised:
            runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.modules[module_name] = imported_module

    assert raised.value.code == 0
    wire = json.loads(capsys.readouterr().out)
    assert wire["schema"] == "studio.differentiation-evidence.v1"


def test_release_profile_and_studio_preserve_the_same_falsification(tmp_path: Path) -> None:
    """A real public facade-to-Studio chain preserves separate verdict axes."""
    from scpn_quantum_control import build_differentiable_release_profile
    from scpn_quantum_control.studio.scorecard_bundle import build_scorecard_bundle

    _, receipt, payload = build_receipt(tmp_path)
    payload["scientific_status"] = "falsified"
    digest = seal(receipt, payload)
    refs = ((receipt, digest),)
    profile = build_differentiable_release_profile(refs, repo_root=VECTOR_ROOT)
    category = next(
        row for row in profile["support_rows"] if row["category"] == payload["category"]
    )
    assert category["domains"][0]["scientific_status"] == "falsified"
    assert category["qualification_status"] == "qualified"
    assert not profile["baseline_release_ready"]
    assert any(
        "scientific claim falsified" in reason
        for row in profile["excluded_capabilities"]
        for reason in row["reasons"]
    )
    bundle = build_scorecard_bundle(qualification_receipts=refs)
    cases = {case.operation_family: case.status for case in bundle.cases}
    assert cases["domain:resource_admission_contract_vector:science:simulation"] == "falsified"
    assert cases["domain:resource_admission_contract_vector:forward"] == "passed"
    assert any(edge.entity_digest == f"sha256:{digest}" for edge in bundle.derived_from)
    assert all(
        status == "behind_baseline"
        for name, status in cases.items()
        if name.startswith("baseline-category:")
    )
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == digest


def test_real_studio_cli_preserves_scientific_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Actual argparse, emitter and federation admission retain falsification."""
    from scpn_quantum_control.studio.scorecard_bundle import main

    _, receipt, payload = build_receipt(tmp_path)
    payload["scientific_status"] = "falsified"
    digest = seal(receipt, payload)
    assert main(["--qualification-receipt", str(receipt), digest]) == 0
    emitted = json.loads(capsys.readouterr().out)
    science = next(
        case
        for case in emitted["cases"]
        if case["operation_family"]
        == "domain:resource_admission_contract_vector:science:simulation"
    )
    assert science["status"] == "falsified"
    assert emitted["claim_boundary"]["status"] == "bounded-model"
