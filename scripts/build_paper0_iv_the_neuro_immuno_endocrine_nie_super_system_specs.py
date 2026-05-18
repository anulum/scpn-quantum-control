#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 IV. The Neuro-Immuno-Endocrine (NIE) Super-System spec builder
"""Promote Paper 0 IV. The Neuro-Immuno-Endocrine (NIE) Super-System records."""

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
    "P0R04921",
    "P0R04922",
    "P0R04923",
    "P0R04924",
    "P0R04925",
    "P0R04926",
    "P0R04927",
    "P0R04928",
    "P0R04929",
    "P0R04930",
    "P0R04931",
    "P0R04932",
    "P0R04933",
    "P0R04934",
)
CLAIM_BOUNDARY = "source-bounded iv the neuro immuno endocrine nie super system source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "iv_the_neuro_immuno_endocrine_nie_super_system.iv_the_neuro_immuno_endocrine_nie_super_system": {
        "context_id": "iv_the_neuro_immuno_endocrine_nie_super_system",
        "validation_protocol": "paper0.iv_the_neuro_immuno_endocrine_nie_super_system.iv_the_neuro_immuno_endocrine_nie_super_system",
        "canonical_statement": "The source-bounded component 'IV. The Neuro-Immuno-Endocrine (NIE) Super-System' preserves Paper 0 records P0R04921-P0R04925 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04921:iv_the_neuro_immuno_endocrine_nie_super_system",
            "P0R04922:iv_the_neuro_immuno_endocrine_nie_super_system",
            "P0R04923:iv_the_neuro_immuno_endocrine_nie_super_system",
            "P0R04924:iv_the_neuro_immuno_endocrine_nie_super_system",
            "P0R04925:iv_the_neuro_immuno_endocrine_nie_super_system",
        ),
        "source_formulae": (
            "P0R04921: IV. The Neuro-Immuno-Endocrine (NIE) Super-System",
            "P0R04922: The NIE system integrates the nervous, immune, and endocrine systems into a unified network responsible for homeostasis and adaptation.",
            "P0R04923: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04924: Fig.: PNI axis: L5 state immune tone; inflammation (IL-1, TNF-) acts as a Decoherence Field (L1 damage; L2/L4 E/I disruption). HPA axis under chronic stress -> allostatic load, L3 atrophy, altered L2 sensitivity, reduced L5 integration.",
            "P0R04925: P0R04925",
        ),
        "test_protocols": (
            "preserve IV. The Neuro-Immuno-Endocrine (NIE) Super-System source-accounting boundary",
        ),
        "null_results": (
            "IV. The Neuro-Immuno-Endocrine (NIE) Super-System is not empirical validation evidence",
        ),
        "variables": ("iv_the_neuro_immuno_endocrine_nie_super_system",),
        "validation_targets": ("preserve records P0R04921-P0R04925",),
        "null_controls": (
            "iv_the_neuro_immuno_endocrine_nie_super_system must remain source-bounded accounting",
        ),
    },
    "iv_the_neuro_immuno_endocrine_nie_super_system.1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi": {
        "context_id": "1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
        "validation_protocol": "paper0.iv_the_neuro_immuno_endocrine_nie_super_system.1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
        "canonical_statement": "The source-bounded component '1. The Psychoneuroimmunology (PNI) Axis and Inflammation (The Decoherence Field)' preserves Paper 0 records P0R04926-P0R04934 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04926:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04927:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04928:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04929:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04930:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04931:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04932:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04933:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
            "P0R04934:1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
        ),
        "source_formulae": (
            "P0R04926: 1. The Psychoneuroimmunology (PNI) Axis and Inflammation (The Decoherence Field)",
            "P0R04927: The PNI axis links mental state (L5) to immune function.",
            "P0R04928: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04929: Fig.: PNI linkage from L5 to inflammation. L5 state -> PNI crosstalk -> pro-inflammatory cytokines (IL-1, TNF-) as a Decoherence Field. Downstream: L1 disruption (oxidative stress -> MTs/mitochondria; QEC impaired) and L2/L4 disruption (NT metabolism shift; E/I imbalance -> sickness/depression), raising F and degrading criticality.",
            'P0R04930: Inflammation as Decoherence: Pro-inflammatory cytokines (e.g., IL-1$\\beta$, TNF-) act as a "Decoherence Field." L1 Disruption: Inflammation increases oxidative stress, damaging the L1 substrate (MTs, Mitochondria) and impairing QEC. | L2/L4 Disruption: Cytokines alter neurotransmitter metabolism and disrupt E/I balance, pushing the network away from criticality (inducing "Sickness Behaviour" or depression).',
            "P0R04931: P0R04931",
            "P0R04932: The Quantum Immune Interface (L1): The Psi-field may directly modulate immune coherence (L1 Quantum Immunology).",
            "P0R04933: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04934: Fig.: Quantum Immune Interface (conceptual). Hypothesis: the Psi-field biases immune coherence at L1, shifting inflammatory tone. Schematic only-mechanistic specifics are theoretical within the SCPN frame.",
        ),
        "test_protocols": (
            "preserve 1. The Psychoneuroimmunology (PNI) Axis and Inflammation (The Decoherence Field) source-accounting boundary",
        ),
        "null_results": (
            "1. The Psychoneuroimmunology (PNI) Axis and Inflammation (The Decoherence Field) is not empirical validation evidence",
        ),
        "variables": ("1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",),
        "validation_targets": ("preserve records P0R04926-P0R04934",),
        "null_controls": (
            "1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IvTheNeuroImmunoEndocrineNieSuperSystemSpec:
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
class IvTheNeuroImmunoEndocrineNieSuperSystemSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IvTheNeuroImmunoEndocrineNieSuperSystemSpec, ...]
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


def build_iv_the_neuro_immuno_endocrine_nie_super_system_specs(
    source_records: list[dict[str, Any]],
) -> IvTheNeuroImmunoEndocrineNieSuperSystemSpecBundle:
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

    specs: list[IvTheNeuroImmunoEndocrineNieSuperSystemSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IvTheNeuroImmunoEndocrineNieSuperSystemSpec(
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
        "title": "Paper 0 " + "IV. The Neuro-Immuno-Endocrine (NIE) Super-System" + " Specs",
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
        "next_source_boundary": "P0R04935",
    }
    return IvTheNeuroImmunoEndocrineNieSuperSystemSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IvTheNeuroImmunoEndocrineNieSuperSystemSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_iv_the_neuro_immuno_endocrine_nie_super_system_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: IvTheNeuroImmunoEndocrineNieSuperSystemSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "IV. The Neuro-Immuno-Endocrine (NIE) Super-System" + " Specs",
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
    bundle: IvTheNeuroImmunoEndocrineNieSuperSystemSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_iv_the_neuro_immuno_endocrine_nie_super_system_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_iv_the_neuro_immuno_endocrine_nie_super_system_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
