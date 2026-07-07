# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio support-matrix bundle tests
"""Tests for the schema-B transform support-matrix bundle emitter (ST-03)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_studio_platform.evidence import EvidenceBundle  # noqa: E402

from scpn_quantum_control.differentiable_transform_algebra import (  # noqa: E402
    run_transform_algebra_audit,
)
from scpn_quantum_control.differentiable_transform_support_matrix import (  # noqa: E402
    REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS,
    TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY,
)
from scpn_quantum_control.studio import support_matrix_bundle  # noqa: E402
from scpn_quantum_control.studio.evidence_bundle import (  # noqa: E402
    StudioBundleValidation,
    validate_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = REPO_ROOT / support_matrix_bundle.DEFAULT_SUPPORT_MATRIX_ARTIFACT_PATH


def test_bundle_is_admitted_by_the_federation_gate() -> None:
    """The emitted bundle passes the platform schema-B validation gate."""
    validated = validate_bundle(support_matrix_bundle.build_support_matrix_bundle())
    assert validated.verdict.admitted, validated.verdict.rejections
    assert validated.bundle.schema == "studio.differentiation-evidence.v1"


def test_bundle_carries_all_rows_with_verbatim_statuses_and_residuals() -> None:
    """All thirteen support rows ride in cases[] untouched."""
    audit = run_transform_algebra_audit()
    bundle = support_matrix_bundle.build_support_matrix_bundle(audit)
    families = [case.operation_family for case in bundle.cases]
    assert families == [
        f"transform-support:{row_id}" for row_id in REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS
    ]
    assert [case.dimension for case in bundle.cases] == list(range(1, 14))
    for case, row in zip(bundle.cases, audit.support_matrix, strict=True):
        assert case.status == row.status
        assert case.error == row.residual
    blocked = [case for case in bundle.cases if case.status == "blocked"]
    assert blocked and all(case.error is None for case in blocked)


def test_bundle_never_upgrades_the_support_matrix_boundary() -> None:
    """The bundle is bounded-model with the matrix's own boundary note."""
    bundle = support_matrix_bundle.build_support_matrix_bundle()
    boundary = bundle.claim_boundary
    assert boundary.status.value == "bounded-model"
    assert boundary.admission.value == "admitted"
    assert boundary.validity_domain is not None
    assert boundary.validity_domain.note == TRANSFORM_ALGEBRA_SUPPORT_MATRIX_CLAIM_BOUNDARY
    wire = json.dumps(bundle.to_dict(), sort_keys=True)
    assert "reference-validated" not in wire
    assert bundle.activity.verb == "differentiate"


def test_bundle_digest_is_deterministic() -> None:
    """Two emissions over deterministic audit replays share one digest."""
    first = support_matrix_bundle.build_support_matrix_bundle()
    second = support_matrix_bundle.build_support_matrix_bundle()
    assert first.entity.digest == second.entity.digest
    assert first.entity.digest.startswith("sha256:")


def test_artifact_edge_content_addresses_the_committed_artefact() -> None:
    """The optional derivation edge digests the committed artefact bytes."""
    bundle = support_matrix_bundle.build_support_matrix_bundle(
        artifact_path=COMMITTED_ARTIFACT,
    )
    assert len(bundle.derived_from) == 1
    edge = bundle.derived_from[0]
    assert edge.entity_digest.startswith("sha256:")
    assert edge.studio == "scpn-quantum-control"
    without = support_matrix_bundle.build_support_matrix_bundle()
    assert without.derived_from == ()


def test_missing_artifact_path_fails_closed(tmp_path: Path) -> None:
    """A requested derivation edge with no artefact refuses to emit."""
    with pytest.raises(ValueError, match="support-matrix artefact does not exist"):
        support_matrix_bundle.build_support_matrix_bundle(
            artifact_path=tmp_path / "missing.json",
        )


def test_failing_audit_is_never_federated() -> None:
    """An audit with a failed case raises instead of emitting."""
    audit = run_transform_algebra_audit()
    first = audit.cases[0]
    tampered_case = dataclasses.replace(
        first,
        status="failed",
        residual=first.tolerance * 2.0,
    )
    tampered = dataclasses.replace(audit, cases=(tampered_case, *audit.cases[1:]))
    with pytest.raises(ValueError, match="audit failed"):
        support_matrix_bundle.build_support_matrix_bundle(tampered)


def test_main_emits_admitted_bundle_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints the wire bundle and exits 0 on admission."""
    exit_code = support_matrix_bundle.main(["--artifact-path", str(COMMITTED_ARTIFACT)])
    captured = capsys.readouterr()
    assert exit_code == 0
    wire = json.loads(captured.out)
    assert wire["schema"] == "studio.differentiation-evidence.v1"
    assert len(wire["cases"]) == 13
    passed_errors = [case["error"] for case in wire["cases"] if case["status"] == "passed"]
    assert passed_errors and all(
        isinstance(error, str) and float(error) >= 0.0 for error in passed_errors
    )
    assert captured.err == ""


def test_main_fails_closed_on_missing_artifact(tmp_path: Path) -> None:
    """The CLI propagates the fail-closed artefact error."""
    with pytest.raises(ValueError, match="support-matrix artefact does not exist"):
        support_matrix_bundle.main(["--artifact-path", str(tmp_path / "missing.json")])


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

    monkeypatch.setattr(support_matrix_bundle, "validate_bundle", rejecting_validate)
    exit_code = support_matrix_bundle.main(["--artifact-path", str(COMMITTED_ARTIFACT)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "forced rejection for the CLI branch" in captured.err
