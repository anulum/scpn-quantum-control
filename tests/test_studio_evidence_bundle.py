# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B EvidenceBundle tests
"""Tests for the QUANTUM Studio schema-B EvidenceBundle emitter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("scpn_studio_platform.evidence", reason="studio extra not installed")

from scpn_studio_platform.evidence import (  # noqa: E402
    AdmissionDecision,
    ClaimStatus,
    EvidenceKind,
    Freshness,
    Substrate,
)

from scpn_quantum_control.differentiable_claim_ledger import ClaimLedgerRow  # noqa: E402
from scpn_quantum_control.hardware_result_packs import (  # noqa: E402
    MANIFEST_RELATIVE_PATH,
    load_manifest,
)
from scpn_quantum_control.studio.evidence_bundle import (  # noqa: E402
    NEGATIVE_RESULT_RULES,
    EvidenceSource,
    build_claim_ledger_bundle,
    build_claim_ledger_bundles,
    build_hardware_result_pack_bundle,
    build_hardware_result_pack_bundles,
    evidence_axes,
    negative_result_evidence,
    validate_bundle,
    validate_bundles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ledger_row(*, claim_id: str, status: str, known_gaps: tuple[str, ...] = ()) -> ClaimLedgerRow:
    """Return a minimal valid claim-ledger row for boundary-contract tests."""
    gaps = known_gaps or ("external validation evidence is not attached",)
    return ClaimLedgerRow(
        claim_id=claim_id,
        claim_text=f"{claim_id} bounded claim",
        implementation_surface=("src/scpn_quantum_control/studio/evidence_bundle.py",),
        test_surface=("tests/test_studio_evidence_bundle.py",),
        docs_surface=("docs/studio_federation.md",),
        evidence_artifact_ids=("artifact",),
        benchmark_artifact_ids=("artifact",),
        known_gaps=gaps,
        promotion_status=status,  # type: ignore[arg-type]
        claim_boundary="synthetic boundary for platform admission mapping",
    )


def test_evidence_axes_declares_schema_b_kind_and_substrate_mapping() -> None:
    """The local source classes map to explicit STUDIO kind and substrate axes."""
    assert evidence_axes("theory") == (EvidenceKind.CURATED, Substrate.CLASSICAL_REFERENCE)
    assert evidence_axes("simulator") == (EvidenceKind.MEASURED, Substrate.SIMULATOR)
    assert evidence_axes("hardware-unmitigated") == (
        EvidenceKind.MEASURED,
        Substrate.HARDWARE_UNMITIGATED,
    )
    assert evidence_axes("hardware-mitigated") == (
        EvidenceKind.HARDWARE_VALIDATED,
        Substrate.HARDWARE_MITIGATED,
    )
    assert evidence_axes("falsification") == (EvidenceKind.FALSIFIED, Substrate.NUMERICAL_MODEL)
    assert evidence_axes("noise-floor") == (
        EvidenceKind.NOISE_LIMITED,
        Substrate.HARDWARE_UNMITIGATED,
    )
    with pytest.raises(ValueError, match="unknown evidence source"):
        evidence_axes(cast(EvidenceSource, "unknown"))


def test_committed_claim_ledger_bundles_validate_through_platform() -> None:
    """The committed differentiable ledger emits admitted, bounded schema-B bundles."""
    bundles = build_claim_ledger_bundles()
    validations = validate_bundles(bundles)

    assert len(validations) == 16
    assert all(validation.verdict.admitted for validation in validations)
    assert all(validation.verdict.rejections == () for validation in validations)
    assert {bundle.schema for bundle in bundles} == {"studio.evidence-replay.v1"}
    assert {bundle.evidence_kind for bundle in bundles} == {EvidenceKind.CURATED}
    assert {bundle.substrate for bundle in bundles} == {Substrate.NUMERICAL_MODEL}
    assert {bundle.claim_boundary.status for bundle in bundles} == {ClaimStatus.BOUNDED_MODEL}


def test_reference_validated_candidate_fails_closed() -> None:
    """Only promoted rows may be elevated to reference-validated status."""
    first_candidate = build_claim_ledger_bundles()[0].entity.entity_id.rsplit(":", maxsplit=1)[-1]

    with pytest.raises(ValueError, match="only admissible for a promoted claim"):
        build_claim_ledger_bundles(reference_validated_claim_ids=(first_candidate,))


def test_promoted_claim_can_emit_reference_validated_boundary() -> None:
    """A promoted row with external certification becomes reference-validated."""
    bundle = build_claim_ledger_bundle(
        _ledger_row(claim_id="promoted-reference", status="promoted"),
        reference_validated=True,
    )
    validation = validate_bundle(bundle)

    assert bundle.claim_boundary.status is ClaimStatus.REFERENCE_VALIDATED
    assert bundle.freshness is Freshness.VERIFIED_AT_SOURCE
    assert validation.verdict.admitted
    assert validation.verdict.rejections == ()


def test_blocked_claim_carries_fail_closed_upstream_dependency() -> None:
    """Blocked rows retain explicit upstream blockers and remain non-admitted."""
    bundle = build_claim_ledger_bundle(
        _ledger_row(
            claim_id="blocked-upstream",
            status="blocked",
            known_gaps=("isolated benchmark runner artefact",),
        )
    )
    validation = validate_bundle(bundle)

    assert bundle.claim_boundary.status is ClaimStatus.EXTERNAL_DEPENDENCY_BLOCKED
    assert tuple(blocker.dependency for blocker in bundle.claim_boundary.blocked_on) == (
        "isolated benchmark runner artefact",
    )
    assert bundle.claim_boundary.admission is AdmissionDecision.REJECTED
    assert validation.verdict.admitted
    assert validation.verdict.mode == "boundary"


def test_hardware_result_pack_bundles_use_physical_substrate_and_artifact_edges() -> None:
    """Committed packs emit bounded support, and the promoted negative pack stays refuted."""
    manifest_path = REPO_ROOT / MANIFEST_RELATIVE_PATH
    manifest = load_manifest(manifest_path)
    bundles = build_hardware_result_pack_bundles(manifest_path=manifest_path)
    validations = validate_bundles(bundles)
    pairs = zip(manifest["packs"], bundles, strict=True)
    by_status = {pack["status"]: bundle for pack, bundle in pairs}
    negative = by_status["promoted_negative_result"]
    positive = [bundle for bundle in bundles if bundle is not negative]

    assert len(bundles) == len(manifest["packs"]) == 5
    assert all(validation.verdict.admitted for validation in validations)
    assert {bundle.schema for bundle in bundles} == {"studio.hardware-result-pack.v1"}
    assert {bundle.substrate for bundle in bundles} == {Substrate.HARDWARE_UNMITIGATED}
    assert {bundle.evidence_kind for bundle in positive} == {EvidenceKind.MEASURED}
    assert {bundle.claim_boundary.status for bundle in positive} == {ClaimStatus.BOUNDED_SUPPORT}
    assert negative.evidence_kind is EvidenceKind.FALSIFIED
    assert negative.claim_boundary.status is ClaimStatus.REFUTED
    assert negative.statistical_evidence is not None
    assert negative.statistical_evidence.to_dict() == {
        "trials": 15,
        "failures": 14,
        "test_name": "Welch t-test per comparison",
    }
    assert {case.status for case in negative.cases} == {"refuted"}
    for pack, bundle in zip(manifest["packs"], bundles, strict=True):
        assert len(bundle.derived_from) == len(pack["artifacts"])
        assert {edge.entity_digest for edge in bundle.derived_from} == {
            f"sha256:{artifact['sha256']}" for artifact in pack["artifacts"]
        }


def test_hardware_bundle_requires_pack_id() -> None:
    """Hardware bundle emission fails closed when a pack ID is missing."""
    with pytest.raises(ValueError, match="non-empty id"):
        build_hardware_result_pack_bundle({"artifacts": []})


def test_hardware_bundle_rejects_artifact_shape_without_inventing_edges() -> None:
    """Malformed artifact collections produce no fabricated cases or digests."""
    scalar_artifacts = build_hardware_result_pack_bundle(
        {"id": "scalar-artifacts", "artifacts": "invalid"}
    )
    mixed_artifacts = build_hardware_result_pack_bundle(
        {
            "id": "mixed-artifacts",
            "artifacts": [
                3,
                {"bytes": True},
                {"role": "text-size", "bytes": "3", "sha256": ""},
            ],
        }
    )

    assert scalar_artifacts.cases == ()
    assert scalar_artifacts.derived_from == ()
    assert len(mixed_artifacts.cases) == 2
    assert {case.dimension for case in mixed_artifacts.cases} == {0}
    assert mixed_artifacts.derived_from == ()


def test_hardware_manifest_bundle_builder_filters_non_mapping_rows(tmp_path: Path) -> None:
    """The manifest route ignores non-object rows after top-level shape validation."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "packs": [3, {"id": "valid-pack", "artifacts": []}],
            }
        ),
        encoding="utf-8",
    )

    bundles = build_hardware_result_pack_bundles(manifest_path=manifest_path)

    assert len(bundles) == 1
    assert bundles[0].entity.entity_id.endswith(":valid-pack")
    assert len(build_hardware_result_pack_bundles()) == 5


def test_validation_summary_is_json_ready() -> None:
    """Validation summaries expose the fields dashboards need without platform objects."""
    validation = validate_bundle(
        build_hardware_result_pack_bundles(manifest_path=REPO_ROOT / MANIFEST_RELATIVE_PATH)[0]
    )

    summary = validation.to_dict()
    assert summary["schema"] == "studio.hardware-result-pack.v1"
    assert summary["entity_id"].startswith("scpn-quantum-control:hardware-result-pack:")
    assert summary["admitted"] is True
    assert summary["rejections"] == []


def _analysis(root: Path, comparisons: object) -> Path:
    """Write a fixture analysis below ``root`` and return the repository root.

    Parameters
    ----------
    root : Path
        Fixture repository root.
    comparisons : object
        The analysis ``comparisons`` value.

    Returns
    -------
    Path
        The fixture repository root.

    """
    rule = NEGATIVE_RESULT_RULES["scpn_fim_negative_ibm_kingston_2026_05_05"]
    path = root / rule.analysis
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"comparisons": comparisons}), encoding="utf-8")
    return root


def test_negative_result_counts_follow_the_declared_rule(tmp_path: Path) -> None:
    """Each comparison of the observable is a trial; each positive delta is a failure.

    Parameters
    ----------
    tmp_path : Path
        Fixture repository root.

    """
    rule = NEGATIVE_RESULT_RULES["scpn_fim_negative_ibm_kingston_2026_05_05"]
    comparisons = [
        {"observable": "magnetisation_leakage", "delta_lambda_4_minus_0": 0.2},
        {"observable": "magnetisation_leakage", "delta_lambda_4_minus_0": -0.1},
        {"observable": "magnetisation_leakage", "delta_lambda_4_minus_0": 0},
        {"observable": "state_retention", "delta_lambda_4_minus_0": 0.5},
        "not a comparison",
    ]

    evidence = negative_result_evidence(rule, _analysis(tmp_path, comparisons))

    assert evidence.to_dict() == {"trials": 3, "failures": 1, "test_name": rule.test_name}


@pytest.mark.parametrize(
    "comparisons",
    [
        None,
        [],
        [{"observable": "magnetisation_leakage", "delta_lambda_4_minus_0": "0.2"}],
        [{"observable": "magnetisation_leakage", "delta_lambda_4_minus_0": True}],
        [{"observable": "magnetisation_leakage", "delta_lambda_4_minus_0": float("nan")}],
    ],
)
def test_negative_result_counts_refuse_an_unusable_analysis(
    tmp_path: Path, comparisons: object
) -> None:
    """No comparison, or a delta that is not a finite number, refuses the count.

    Parameters
    ----------
    tmp_path : Path
        Fixture repository root.
    comparisons : object
        The analysis ``comparisons`` value.

    """
    rule = NEGATIVE_RESULT_RULES["scpn_fim_negative_ibm_kingston_2026_05_05"]
    with pytest.raises(ValueError, match="needs finite"):
        negative_result_evidence(rule, _analysis(tmp_path, comparisons))


def test_a_negative_pack_without_a_rule_refuses_emission() -> None:
    """A promoted negative pack with no declared rule never falls back to bounded support."""
    with pytest.raises(ValueError, match="needs a declared counting rule"):
        build_hardware_result_pack_bundle(
            {"id": "undeclared", "status": "promoted_negative_result", "artifacts": []}
        )


def test_a_non_object_analysis_refuses_the_count(tmp_path: Path) -> None:
    """An analysis file that is not a JSON object holds no comparisons.

    Parameters
    ----------
    tmp_path : Path
        Fixture repository root.

    """
    rule = NEGATIVE_RESULT_RULES["scpn_fim_negative_ibm_kingston_2026_05_05"]
    path = tmp_path / rule.analysis
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="needs finite"):
        negative_result_evidence(rule, tmp_path)


@pytest.mark.parametrize(
    "artifacts",
    [
        [],
        "invalid",
        [
            {
                "role": "raw_counts",
                "path": NEGATIVE_RESULT_RULES[
                    "scpn_fim_negative_ibm_kingston_2026_05_05"
                ].analysis,
                "sha256": "a" * 64,
            }
        ],
        [
            {
                "role": "reproduced_analysis",
                "path": NEGATIVE_RESULT_RULES[
                    "scpn_fim_negative_ibm_kingston_2026_05_05"
                ].analysis,
                "sha256": "",
            }
        ],
    ],
)
def test_a_negative_pack_must_record_its_declared_analysis(artifacts: object) -> None:
    """The counted analysis must be a reproduced analysis the pack itself records.

    Parameters
    ----------
    artifacts : object
        The pack's ``artifacts`` value.

    """
    with pytest.raises(ValueError, match="is not a recorded reproduced analysis"):
        build_hardware_result_pack_bundle(
            {
                "id": "scpn_fim_negative_ibm_kingston_2026_05_05",
                "status": "promoted_negative_result",
                "artifacts": artifacts,
            }
        )


def test_negative_result_counts_only_the_recorded_bytes(tmp_path: Path) -> None:
    """Counting refuses analysis bytes that differ from the digest the pack records.

    Parameters
    ----------
    tmp_path : Path
        Fixture repository root.

    """
    rule = NEGATIVE_RESULT_RULES["scpn_fim_negative_ibm_kingston_2026_05_05"]
    comparisons = [{"observable": "magnetisation_leakage", "delta_lambda_4_minus_0": 0.2}]
    root = _analysis(tmp_path, comparisons)
    recorded = hashlib.sha256((root / rule.analysis).read_bytes()).hexdigest()

    counted = negative_result_evidence(rule, root, expected_sha256=recorded)

    assert (counted.trials, counted.failures) == (1, 1)
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        negative_result_evidence(rule, root, expected_sha256="0" * 64)
