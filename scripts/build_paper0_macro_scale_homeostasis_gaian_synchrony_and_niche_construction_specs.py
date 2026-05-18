#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction spec builder
"""Promote Paper 0 Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction records."""

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
    "P0R05528",
    "P0R05529",
    "P0R05530",
    "P0R05531",
    "P0R05532",
    "P0R05533",
    "P0R05534",
    "P0R05535",
    "P0R05536",
)
CLAIM_BOUNDARY = "source-bounded macro scale homeostasis gaian synchrony and niche construction source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "macro_scale_homeostasis_gaian_synchrony_and_niche_construction.macro_scale_homeostasis_gaian_synchrony_and_niche_construction": {
        "context_id": "macro_scale_homeostasis_gaian_synchrony_and_niche_construction",
        "validation_protocol": "paper0.macro_scale_homeostasis_gaian_synchrony_and_niche_construction.macro_scale_homeostasis_gaian_synchrony_and_niche_construction",
        "canonical_statement": "The source-bounded component 'Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction' preserves Paper 0 records P0R05528-P0R05531 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05528:macro_scale_homeostasis_gaian_synchrony_and_niche_construction",
            "P0R05529:macro_scale_homeostasis_gaian_synchrony_and_niche_construction",
            "P0R05530:macro_scale_homeostasis_gaian_synchrony_and_niche_construction",
            "P0R05531:macro_scale_homeostasis_gaian_synchrony_and_niche_construction",
        ),
        "source_formulae": (
            "P0R05528: Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction",
            'P0R05529: This principle of active environmental engineering is not confined to the micro-scale of the brain. The SCPN\'s Layer 12, "Ecological-Gaian Synchrony," describes the entire biosphere as a single, integrated super-organism with homeostatic feedback loops that maintain planetary stability, a concept directly inspired by the Gaia hypothesis. This macroscopic homeostasis can be formalised through the lens of Niche Construction Theory (NCT) and the Free Energy Principle (FEP).',
            "P0R05530: NCT posits that organisms are not passive recipients of environmental pressures but are active constructors of their own ecological niches. They modify their environment through their metabolic and behavioural activities, thereby altering the selection pressures they and other species face. When framed by the FEP, niche construction becomes a process of collective active inference. The collective of organisms acts to modify its shared environment to make its sensory inputs more predictable and congruent with its implicit generative model of a viable world, thereby minimising its collective free energy.",
            "P0R05531: The homeostatic loops of the Gaian field described in Layer 12 are the planetary-scale result of this collective niche construction. For example, the regulation of atmospheric oxygen and carbon dioxide by the collective metabolism of phytoplankton and forests is a physical manifestation of a planetary-scale agent acting on its environment to maintain the conditions necessary for its own existence. Biodiversity, in this context, plays a critical role as a stabiliser, providing the redundancy and complexity necessary for this global regulatory network to be resilient to perturbations.",
        ),
        "test_protocols": (
            "preserve Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction source-accounting boundary",
        ),
        "null_results": (
            "Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction is not empirical validation evidence",
        ),
        "variables": ("macro_scale_homeostasis_gaian_synchrony_and_niche_construction",),
        "validation_targets": ("preserve records P0R05528-P0R05531",),
        "null_controls": (
            "macro_scale_homeostasis_gaian_synchrony_and_niche_construction must remain source-bounded accounting",
        ),
    },
    "macro_scale_homeostasis_gaian_synchrony_and_niche_construction.a_scale_invariant_principle_of_active_homeostasis": {
        "context_id": "a_scale_invariant_principle_of_active_homeostasis",
        "validation_protocol": "paper0.macro_scale_homeostasis_gaian_synchrony_and_niche_construction.a_scale_invariant_principle_of_active_homeostasis",
        "canonical_statement": "The source-bounded component 'A Scale-Invariant Principle of Active Homeostasis' preserves Paper 0 records P0R05532-P0R05536 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05532:a_scale_invariant_principle_of_active_homeostasis",
            "P0R05533:a_scale_invariant_principle_of_active_homeostasis",
            "P0R05534:a_scale_invariant_principle_of_active_homeostasis",
            "P0R05535:a_scale_invariant_principle_of_active_homeostasis",
            "P0R05536:a_scale_invariant_principle_of_active_homeostasis",
        ),
        "source_formulae": (
            "P0R05532: A Scale-Invariant Principle of Active Homeostasis",
            "P0R05533: The parallel between glial control of the neuronal niche and Gaian control of the planetary niche reveals a deep, scale-invariant principle. Biological systems across all scales exhibit fractal geometry and scale-free dynamics, from the power-law distribution of neuronal avalanches to the scale-free topology of metabolic and ecological networks.",
            "P0R05534: This suggests that the cybernetic logic of life is itself fractal. The process of active, hierarchical control of a niche to maintain the conditions for existence is a fundamental principle that repeats at every level of organisation.",
            "P0R05535: The glial network's regulation of its internal neuronal environment and the biosphere's regulation of its external planetary environment are not merely analogous; they are different-scale manifestations of the same universal imperative described by the FEP.",
            "P0R05536: This provides a continuous theoretical thread that connects the biophysics of a single brain to the ecology of the entire planet, revealing a unified principle of self-organising, coherence-maintaining dynamics across all scales of life.",
        ),
        "test_protocols": (
            "preserve A Scale-Invariant Principle of Active Homeostasis source-accounting boundary",
        ),
        "null_results": (
            "A Scale-Invariant Principle of Active Homeostasis is not empirical validation evidence",
        ),
        "variables": ("a_scale_invariant_principle_of_active_homeostasis",),
        "validation_targets": ("preserve records P0R05532-P0R05536",),
        "null_controls": (
            "a_scale_invariant_principle_of_active_homeostasis must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpec:
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
class MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpec, ...]
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


def build_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_specs(
    source_records: list[dict[str, Any]],
) -> MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpecBundle:
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

    specs: list[MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpec(
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
        + "Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction"
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
        "next_source_boundary": "P0R05537",
    }
    return MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_specs(
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
    bundle: MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction"
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
    bundle: MacroScaleHomeostasisGaianSynchronyAndNicheConstructionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_validation_specs_{date_tag}.md"
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
