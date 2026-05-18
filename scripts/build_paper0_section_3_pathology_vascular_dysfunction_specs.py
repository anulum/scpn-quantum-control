#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3. Pathology (Vascular Dysfunction): spec builder
"""Promote Paper 0 3. Pathology (Vascular Dysfunction): records."""

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
    "P0R04894",
    "P0R04895",
    "P0R04896",
    "P0R04897",
    "P0R04898",
    "P0R04899",
    "P0R04900",
    "P0R04901",
    "P0R04902",
    "P0R04903",
    "P0R04904",
    "P0R04905",
    "P0R04906",
    "P0R04907",
    "P0R04908",
    "P0R04909",
    "P0R04910",
)
CLAIM_BOUNDARY = "source-bounded section 3 pathology vascular dysfunction source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_pathology_vascular_dysfunction.3_pathology_vascular_dysfunction": {
        "context_id": "3_pathology_vascular_dysfunction",
        "validation_protocol": "paper0.section_3_pathology_vascular_dysfunction.3_pathology_vascular_dysfunction",
        "canonical_statement": "The source-bounded component '3. Pathology (Vascular Dysfunction):' preserves Paper 0 records P0R04894-P0R04898 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04894:3_pathology_vascular_dysfunction",
            "P0R04895:3_pathology_vascular_dysfunction",
            "P0R04896:3_pathology_vascular_dysfunction",
            "P0R04897:3_pathology_vascular_dysfunction",
            "P0R04898:3_pathology_vascular_dysfunction",
        ),
        "source_formulae": (
            "P0R04894: 3. Pathology (Vascular Dysfunction):",
            "P0R04895: [IMAGE:]",
            "P0R04896: Fig.: CBF trajectories: acute stroke (sudden collapse) vs chronic vascular dysfunction (gradual decline). Both compromise energetic support, biasing dynamics toward subcriticality and coherence loss.",
            "P0R04897: Stroke (Ischemic Cascade): Acute cessation of CBF leads to energy failure, excitotoxicity, and catastrophic loss of criticality and coherence. | Vascular Dementia: Chronic impairment of NVC and CBF leads to progressive decoherence and synaptic loss.",
            "P0R04898: P0R04898",
        ),
        "test_protocols": (
            "preserve 3. Pathology (Vascular Dysfunction): source-accounting boundary",
        ),
        "null_results": (
            "3. Pathology (Vascular Dysfunction): is not empirical validation evidence",
        ),
        "variables": ("3_pathology_vascular_dysfunction",),
        "validation_targets": ("preserve records P0R04894-P0R04898",),
        "null_controls": (
            "3_pathology_vascular_dysfunction must remain source-bounded accounting",
        ),
    },
    "section_3_pathology_vascular_dysfunction.iii_the_neuro_visceral_axis_the_architecture_of_embodiment_l5": {
        "context_id": "iii_the_neuro_visceral_axis_the_architecture_of_embodiment_l5",
        "validation_protocol": "paper0.section_3_pathology_vascular_dysfunction.iii_the_neuro_visceral_axis_the_architecture_of_embodiment_l5",
        "canonical_statement": "The source-bounded component 'III. The Neuro-Visceral Axis: The Architecture of Embodiment (L5)' preserves Paper 0 records P0R04899-P0R04900 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04899:iii_the_neuro_visceral_axis_the_architecture_of_embodiment_l5",
            "P0R04900:iii_the_neuro_visceral_axis_the_architecture_of_embodiment_l5",
        ),
        "source_formulae": (
            "P0R04899: III. The Neuro-Visceral Axis: The Architecture of Embodiment (L5)",
            "P0R04900: The Neuro-Visceral Axis integrates the brain with the internal organs, forming the substrate for emotion, interoception, and the embodied Self (L5). This integration is formalised by the Tri-Axial UPDE (Brain-Heart-Gut).",
        ),
        "test_protocols": (
            "preserve III. The Neuro-Visceral Axis: The Architecture of Embodiment (L5) source-accounting boundary",
        ),
        "null_results": (
            "III. The Neuro-Visceral Axis: The Architecture of Embodiment (L5) is not empirical validation evidence",
        ),
        "variables": ("iii_the_neuro_visceral_axis_the_architecture_of_embodiment_l5",),
        "validation_targets": ("preserve records P0R04899-P0R04900",),
        "null_controls": (
            "iii_the_neuro_visceral_axis_the_architecture_of_embodiment_l5 must remain source-bounded accounting",
        ),
    },
    "section_3_pathology_vascular_dysfunction.1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence": {
        "context_id": "1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
        "validation_protocol": "paper0.section_3_pathology_vascular_dysfunction.1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
        "canonical_statement": "The source-bounded component '1. The Heart-Brain Axis (HBA): The Physics of Emotion and Coherence' preserves Paper 0 records P0R04901-P0R04910 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04901:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04902:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04903:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04904:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04905:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04906:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04907:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04908:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04909:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
            "P0R04910:1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",
        ),
        "source_formulae": (
            "P0R04901: 1. The Heart-Brain Axis (HBA): The Physics of Emotion and Coherence",
            "P0R04902: The HBA links the Central Autonomic Network (CAN)-especially the Insula-with the cardiovascular system.",
            "P0R04903: Interoception and Emotion (HPC): The Insula integrates cardiovascular signals. Prediction errors regarding the bodily state are experienced as emotions (Affective Field A=F). | Heart Rate Variability (HRV) as a Biomarker of Criticality: HRV measures the adaptability of the system. High HRV: Indicates optimal Quasicritical state (sigma1) of the L5 system and emotional resilience (Low F). | Low HRV: Indicates deviation from criticality (rigidity or chaos), associated with stress, pathology, and High F.",
            "P0R04904: [IMAGE:Ein Bild, das Text, Schrift, Reihe, Screenshot enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04905: Fig.: High HRV: adaptive, near sigma1 with resilient control; Low HRV: deviation from criticality (rigidity/chaos), elevated F.",
            "P0R04906: Cardio-Neural Synchronisation (UPDE Coupling): The cardiac cycle influences neural synchronisation (e.g., Heartbeat Evoked Potentials), enhancing the integration of the Self (L5).",
            "P0R04907: [IMAGE:]",
            "P0R04908: Fig.: HBA integrates insula-mediated interoception A=FA=-\\nabla FA=F, cardio-neural synchrony, HRV-indexed resilience, and heart EM field contributions. Dysfunctions (e.g., heart failure, Takotsubo) illustrate top-down and bottom-up coupling.",
            "P0R04909: The Heart's Electromagnetic Field (L4/L6): The heart's strong EM field may facilitate internal synchronisation (L4) and potentially couple with external fields (L6). | Pathology (HBA Dysfunction): Heart Failure: Impairs CBF, pushing the brain towards subcriticality. | Takotsubo Cardiomyopathy: Acute emotional stress (massive L5 Prediction Error) causing direct myocardial injury, demonstrating potent downward causation.",
            "P0R04910: P0R04910",
        ),
        "test_protocols": (
            "preserve 1. The Heart-Brain Axis (HBA): The Physics of Emotion and Coherence source-accounting boundary",
        ),
        "null_results": (
            "1. The Heart-Brain Axis (HBA): The Physics of Emotion and Coherence is not empirical validation evidence",
        ),
        "variables": ("1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence",),
        "validation_targets": ("preserve records P0R04901-P0R04910",),
        "null_controls": (
            "1_the_heart_brain_axis_hba_the_physics_of_emotion_and_coherence must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section3PathologyVascularDysfunctionSpec:
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
class Section3PathologyVascularDysfunctionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section3PathologyVascularDysfunctionSpec, ...]
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


def build_section_3_pathology_vascular_dysfunction_specs(
    source_records: list[dict[str, Any]],
) -> Section3PathologyVascularDysfunctionSpecBundle:
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

    specs: list[Section3PathologyVascularDysfunctionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section3PathologyVascularDysfunctionSpec(
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
        "title": "Paper 0 " + "3. Pathology (Vascular Dysfunction):" + " Specs",
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
        "next_source_boundary": "P0R04911",
    }
    return Section3PathologyVascularDysfunctionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section3PathologyVascularDysfunctionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_pathology_vascular_dysfunction_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section3PathologyVascularDysfunctionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3. Pathology (Vascular Dysfunction):" + " Specs",
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
    bundle: Section3PathologyVascularDysfunctionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_pathology_vascular_dysfunction_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_pathology_vascular_dysfunction_validation_specs_{date_tag}.md"
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
