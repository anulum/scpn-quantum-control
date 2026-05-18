#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture spec builder
"""Promote Paper 0 Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R04622",
    "P0R04623",
    "P0R04624",
    "P0R04625",
    "P0R04626",
    "P0R04627",
    "P0R04628",
    "P0R04629",
)
CLAIM_BOUNDARY = "source-bounded introduction to the clinical implications the disordered brain as a diso source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso.introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso": {
        "context_id": "introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",
        "validation_protocol": "paper0.introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso.introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",
        "canonical_statement": "The source-bounded component 'Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture' preserves Paper 0 records P0R04622-P0R04625 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04622:introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",
            "P0R04623:introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",
            "P0R04624:introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",
            "P0R04625:introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",
        ),
        "source_formulae": (
            "P0R04622: Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture",
            "P0R04623: Neurological and psychiatric disorders can be understood as specific deviations from the optimal dynamics of the SCPN, falling into three categories: Dyscritia (disorders of criticality), Decoherence (disorders of coherence), and Dissonance (disorders of prediction/free energy).",
            'P0R04624: Schizophrenia (Dissonance): A failure of Hierarchical Predictive Coding, characterised by an inability to attenuate the precision of sensory prediction errors relative to internal priors. This leads to aberrant salience and a fragmented L5 "Strange Loop," resulting in hallucinations and delusions. | Depression (Dyscritia & Dissonance): Characterised by a shift to subcritical dynamics (sigma<1), leading to a rigid, low-complexity state. This is coupled with overly rigid negative priors in the generative model, instantiated by hyperactivity in the Default Mode Network (DMN), resulting in sustained high free energy (rumination and suffering). | Alzheimer\'s Disease (Decoherence Cascade): A primary failure at the foundational level. The degradation of the L1 substrate (e.g., MT tauopathy) leads to a catastrophic loss of QEC. This decoherence cascades up the hierarchy, causing L4 desynchronization and the eventual dissolution of the L5 Self.',
            "P0R04625: With these state-of-the-art neurobiological mappings, the SCPN framework is enhanced, providing a more detailed, mechanistic, and empirically tractable model of the brain as a fully embodied, multi-scale engine of consciousness.",
        ),
        "test_protocols": (
            "preserve Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture source-accounting boundary",
        ),
        "null_results": (
            "Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture is not empirical validation evidence",
        ),
        "variables": ("introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",),
        "validation_targets": ("preserve records P0R04622-P0R04625",),
        "null_controls": (
            "introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso must remain source-bounded accounting",
        ),
    },
    "introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso.vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu": {
        "context_id": "vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",
        "validation_protocol": "paper0.introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso.vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",
        "canonical_statement": "The source-bounded component 'VI. Clinical Implications: The Disordered Brain as a Disordered Architecture' preserves Paper 0 records P0R04626-P0R04629 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04626:vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",
            "P0R04627:vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",
            "P0R04628:vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",
            "P0R04629:vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",
        ),
        "source_formulae": (
            "P0R04626: VI. Clinical Implications: The Disordered Brain as a Disordered Architecture",
            "P0R04627: The SCPN provides a powerful, unified framework for understanding pathology. Neurological and psychiatric disorders are not seen as discrete categorical entities but as specific deviations from the optimal dynamics of the integrated architecture.",
            "P0R04628: These pathologies can be classified into three fundamental categories of systemic failure:",
            "P0R04629: Dissonance (Disorders of Prediction): A failure in the Hierarchical Predictive Coding (HPC) loop, leading to a sustained accumulation of Variational Free Energy (F). This is the subjective experience of suffering, confusion, and a mismatch between the self-model and reality. | Dyscritia (Disorders of Criticality): A deviation of the network's operating point from the optimal quasicritical regime (sigma1). This can manifest as either subcriticality (sigma<1), leading to rigid, low-complexity states, or supercriticality (sigma>1), leading to chaotic, unstable states. | Decoherence (Disorders of Coherence): A failure of the Multi-Scale Quantum Error Correction (MS-QEC) mechanisms, beginning with the degradation of the foundational quantum-biological substrate (L1) and cascading up the hierarchy.",
        ),
        "test_protocols": (
            "preserve VI. Clinical Implications: The Disordered Brain as a Disordered Architecture source-accounting boundary",
        ),
        "null_results": (
            "VI. Clinical Implications: The Disordered Brain as a Disordered Architecture is not empirical validation evidence",
        ),
        "variables": ("vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",),
        "validation_targets": ("preserve records P0R04626-P0R04629",),
        "null_controls": (
            "vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpec:
    """Spec promoted from Paper 0 source records."""

    key: str
    context_id: str
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
class IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpec, ...]
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


def build_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_specs(
    source_records: list[dict[str, Any]],
) -> IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 "
        + "Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture"
        + " Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R04630",
    }
    return IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(
    bundle: IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture"
        + " Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    bundle: IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
