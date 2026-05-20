#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 cosmological predictions spec builder
"""Promote Paper 0 five cosmological predictions records into specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6949, 7006))
CLAIM_BOUNDARY = "source-bounded preregistration protocol catalogue; not empirical evidence"
HARDWARE_STATUS = "preregistration_protocol_no_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "cosmological_predictions.chapter_boundary": {
        "prediction_id": "chapter",
        "validation_protocol": "paper0.cosmological_predictions.chapter_boundary",
        "canonical_statement": "Chapter 28 lists five preregisterable cosmological consequences with null results.",
        "source_equation_ids": ("P0R06949:chapter_boundary", "P0R06951:defined_null_result"),
        "source_formulae": (
            "five predictions distinguish the framework from standard Lambda-CDM",
        ),
        "test_protocols": ("require preregistration and defined null result for each prediction",),
        "null_results": ("unsupported confirmation claims are rejected",),
        "variables": ("prediction", "null_result", "preregistration"),
        "validation_targets": (
            "preserve chapter boundary",
            "preserve preregistration requirement",
            "reject treating predictions as confirmed observations",
        ),
        "null_controls": (
            "missing-preregistration control must be rejected",
            "missing-null-result control must be rejected",
            "confirmation-overclaim control must be rejected",
        ),
    },
    "cosmological_predictions.cmb_correlations": {
        "prediction_id": "28.1",
        "validation_protocol": "paper0.cosmological_predictions.cmb_correlations",
        "canonical_statement": "CMB phase correlations or periodic modulations are stated as a blind-template target.",
        "source_equation_ids": ("P0R06954:cmb_prediction", "P0R06956:cmb_test_protocol"),
        "source_formulae": ("non-random phase correlations or periodic modulations",),
        "test_protocols": (
            "pre-register harmonic templates",
            "run blind template matching on Planck and Simons Observatory maps",
            "control false discovery rate",
        ),
        "null_results": (
            "no significant excess above noise floor at declared power threshold",
            "CMB remains consistent with inflation-generated Gaussian random field",
        ),
        "variables": ("CMB", "harmonic_template", "false_discovery_rate"),
        "validation_targets": (
            "preserve CMB prediction target",
            "preserve blind-template matching requirement",
            "preserve explicit null result",
        ),
        "null_controls": (
            "missing-CMB-null-result control must be rejected",
            "unblinded-template-analysis control must be rejected",
            "post-hoc-threshold control must be rejected",
        ),
    },
    "cosmological_predictions.gravitational_wave_sidebands": {
        "prediction_id": "28.2",
        "validation_protocol": "paper0.cosmological_predictions.gravitational_wave_sidebands",
        "canonical_statement": "Post-merger ringdown echoes or sidebands are stated as a residual-analysis target.",
        "source_equation_ids": ("P0R06960:gw_prediction", "P0R06961:gw_test_protocol"),
        "source_formulae": ("echoes, frequency sidebands, or periodic ringdown modulations",),
        "test_protocols": (
            "analyse LIGO/Virgo/KAGRA merger residuals",
            "compare against standard ringdown and periodic microstructure templates",
        ),
        "null_results": ("GR ringdown modes describe all events within declared sensitivity",),
        "variables": ("ringdown", "sideband", "strain_residual"),
        "validation_targets": (
            "preserve gravitational-wave prediction target",
            "preserve residual-analysis requirement",
            "preserve GR-null result",
        ),
        "null_controls": (
            "missing-GW-null-result control must be rejected",
            "missing-standard-GR-template control must be rejected",
            "uncontrolled-sideband-threshold control must be rejected",
        ),
    },
    "cosmological_predictions.observer_entropy_anomaly": {
        "prediction_id": "28.3",
        "validation_protocol": "paper0.cosmological_predictions.observer_entropy_anomaly",
        "canonical_statement": "Observer-linked entropy anomalies are stated as high-risk blinded precision tests.",
        "source_equation_ids": (
            "P0R06967:entropy_prediction",
            "P0R06968:entropy_test_protocol",
            "P0R06971:high_risk_note",
            "P0R06972:small_effect_size",
        ),
        "source_formulae": (
            "lambda_psi_EM approximately 0.092; expected deviation at or below 1e-6",
        ),
        "test_protocols": (
            "monitor isolated thermal or quantum random fluctuation devices",
            "compare observer and non-observer conditions",
            "pre-register effect size, sample size, and analysis pipeline",
            "include phase-randomised sham controls and blinding",
        ),
        "null_results": ("no significant entropy-statistics difference at declared power",),
        "variables": ("lambda_psi_EM", "C_global", "entropy_rate"),
        "validation_targets": (
            "preserve high-risk boundary",
            "preserve blinding and sham-control requirements",
            "preserve no-reproducible-second-law-violation statement",
        ),
        "null_controls": (
            "missing-blinding-control must be rejected",
            "missing-effect-size-preregistration control must be rejected",
            "second-law-violation-overclaim control must be rejected",
        ),
    },
    "cosmological_predictions.arrow_time_palindrome": {
        "prediction_id": "28.4",
        "validation_protocol": "paper0.cosmological_predictions.arrow_time_palindrome",
        "canonical_statement": "Time-symmetric perturbation spectra are stated as a future high-bar target.",
        "source_equation_ids": (
            "P0R06975:palindrome_prediction",
            "P0R06977:palindrome_test_protocol",
        ),
        "source_formulae": ("palindromic structure around the bounce",),
        "test_protocols": (
            "search primordial perturbation spectra for mirror symmetry",
            "compare graviton background against inflation and bounce templates",
        ),
        "null_results": (
            "perturbation spectrum remains consistent with standard slow-roll inflation",
        ),
        "variables": ("perturbation_spectrum", "bounce", "mirror_symmetry"),
        "validation_targets": (
            "preserve arrow-of-time prediction target",
            "preserve inflation-vs-bounce comparison",
            "preserve null result",
        ),
        "null_controls": (
            "missing-inflation-baseline control must be rejected",
            "missing-palindrome-null control must be rejected",
            "future-instrumentation-overclaim control must be rejected",
        ),
    },
    "cosmological_predictions.quantum_information_retention": {
        "prediction_id": "28.5",
        "validation_protocol": "paper0.cosmological_predictions.quantum_information_retention",
        "canonical_statement": "Quantum information retention is stated as partial-revival testing under controlled decoherence.",
        "source_equation_ids": (
            "P0R06983:information_retention_prediction",
            "P0R06985:information_retention_test_protocol",
        ),
        "source_formulae": (
            "small unexplained partial revival or plateau in entanglement entropy",
        ),
        "test_protocols": (
            "prepare maximally entangled states in a well-characterised environment",
            "monitor entanglement entropy during decoherence",
            "distinguish from known environmental recurrences and bath memory effects",
        ),
        "null_results": ("entanglement entropy follows predicted monotonic decoherence curve",),
        "variables": ("entanglement_entropy", "decoherence", "bath_memory"),
        "validation_targets": (
            "preserve quantum information-retention prediction target",
            "preserve known-recurrence controls",
            "preserve monotonic-decoherence null result",
        ),
        "null_controls": (
            "missing-decoherence-null control must be rejected",
            "missing-bath-memory-control must be rejected",
            "partial-revival-overclaim control must be rejected",
        ),
    },
    "cosmological_predictions.cross_prediction_consistency": {
        "prediction_id": "28.6",
        "validation_protocol": "paper0.cosmological_predictions.cross_consistency",
        "canonical_statement": "The five predictions are cross-linked by consistency requirements.",
        "source_equation_ids": (
            "P0R06989:cross_prediction_section",
            "P0R06991:cmb_quantum_retention_tension",
            "P0R06993:entropy_retention_effect_size_link",
            "P0R06994:gw_cmb_frequency_link",
            "P0R06995:abandonment_or_revision_boundary",
        ),
        "source_formulae": (
            "confirmed prediction without cross-linked partner creates internal tension",
        ),
        "test_protocols": ("evaluate cross-linked confirmations and nulls jointly",),
        "null_results": (
            "isolated confirmation requires coupling hierarchy revision or framework revision",
        ),
        "variables": ("P28_1", "P28_2", "P28_3", "P28_5", "coupling_hierarchy"),
        "validation_targets": (
            "preserve cross-consistency rules",
            "preserve internal-tension boundary",
            "reject cherry-picked isolated confirmation",
        ),
        "null_controls": (
            "missing-cross-link control must be rejected",
            "isolated-confirmation-overclaim control must be rejected",
            "coupling-hierarchy-revision-omission control must be rejected",
        ),
    },
    "cosmological_predictions.priority_ranking": {
        "prediction_id": "28.7",
        "validation_protocol": "paper0.cosmological_predictions.priority_ranking",
        "canonical_statement": "The predictions are ranked by current or near-term feasibility.",
        "source_equation_ids": (
            "P0R06998:priority_1",
            "P0R06999:priority_2",
            "P0R07000:priority_3",
            "P0R07001:priority_4",
            "P0R07002:priority_5",
        ),
        "source_formulae": ("28.1, 28.2, 28.5, 28.3, 28.4 feasibility order",),
        "test_protocols": (
            "prioritise Planck CMB, LIGO O4/O5, and current qubit platforms first",
        ),
        "null_results": (
            "priority order is not evidence and may be revised by instrumentation changes",
        ),
        "variables": ("priority", "instrumentation"),
        "validation_targets": (
            "preserve source priority order",
            "preserve instrumentation rationale",
            "reject priority as evidence strength",
        ),
        "null_controls": (
            "invalid-priority-order control must be rejected",
            "priority-as-confirmation control must be rejected",
            "missing-instrumentation-rationale control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CosmologicalPredictionsSpec:
    """Cosmological predictions spec promoted from Paper 0 records."""

    key: str
    prediction_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class CosmologicalPredictionsSpecBundle:
    """Cosmological predictions specs plus coverage summary."""

    specs: tuple[CosmologicalPredictionsSpec, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL ledger into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_cosmological_predictions_specs(
    source_records: list[dict[str, Any]],
) -> CosmologicalPredictionsSpecBundle:
    """Build source-covered cosmological predictions specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CosmologicalPredictionsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CosmologicalPredictionsSpec(
                key=key,
                prediction_id=str(metadata["prediction_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Cosmological Predictions Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "prediction_count": 5,
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_null_results": all(bool(spec.null_results) for spec in specs),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return CosmologicalPredictionsSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: CosmologicalPredictionsSpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Cosmological Predictions Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Prediction count: {bundle.summary['prediction_count']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                f"- Prediction: {spec.prediction_id}",
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Null results: {len(spec.null_results)}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: CosmologicalPredictionsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the cosmological predictions specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_cosmological_predictions_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_cosmological_predictions_validation_specs_report_{date_tag}.md"
    )
    payload = {"specs": [asdict(spec) for spec in bundle.specs], "summary": bundle.summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build cosmological predictions specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_cosmological_predictions_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
