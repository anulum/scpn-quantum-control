#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Scale-Invariant Cybernetic Principle spec builder
"""Promote Paper 0 Scale-Invariant Cybernetic Principle records."""

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
    "P0R05493",
    "P0R05494",
    "P0R05495",
    "P0R05496",
    "P0R05497",
    "P0R05498",
    "P0R05499",
    "P0R05500",
    "P0R05501",
    "P0R05502",
    "P0R05503",
    "P0R05504",
    "P0R05505",
    "P0R05506",
    "P0R05507",
)
CLAIM_BOUNDARY = "source-bounded scale invariant cybernetic principle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "scale_invariant_cybernetic_principle.scale_invariant_cybernetic_principle": {
        "context_id": "scale_invariant_cybernetic_principle",
        "validation_protocol": "paper0.scale_invariant_cybernetic_principle.scale_invariant_cybernetic_principle",
        "canonical_statement": "The source-bounded component 'Scale-Invariant Cybernetic Principle' preserves Paper 0 records P0R05493-P0R05496 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05493:scale_invariant_cybernetic_principle",
            "P0R05494:scale_invariant_cybernetic_principle",
            "P0R05495:scale_invariant_cybernetic_principle",
            "P0R05496:scale_invariant_cybernetic_principle",
        ),
        "source_formulae": (
            "P0R05493: Scale-Invariant Cybernetic Principle",
            'P0R05494: The relationship between the fast neuronal network and the slow glial control network is not a unique feature of the brain but is a specific instantiation of a universal, scale-invariant cybernetic principle that repeats across the entire SCPN architecture. At the neural scale (Layer 4), the glial network actively engineers the local ionic and neurochemical environment-its "niche"-to ensure the optimal computational function of the neuronal network. This is a form of internal niche construction.',
            'P0R05495: This same logic applies at the planetary scale. The manuscript\'s Layer 12, "Ecological-Gaian Synchrony," describes the entire biosphere as a single, integrated system with homeostatic feedback loops that maintain planetary stability. This macroscopic homeostasis is the result of collective niche construction on a planetary scale. The collective of all living organisms (the "fast" system) acts to modify its shared environment (the "slow" system) to maintain the conditions necessary for its own existence (e.g., regulation of atmospheric gases).',
            "P0R05496: This parallel reveals a deep, fractal logic within the SCPN. The relationship (Glia : Neuron) is functionally and mathematically homologous to the relationship (Gaia : Species). This allows for the formulation of a Scale-Invariant Homeostasis Lemma: Any fast computational substrate within the SCPN requires a slower, integrative modulatory layer to maintain its dynamic stability near a critical or quasicritical optimum. This principle provides a powerful unifying thread that connects the biophysics of a single brain (Domain I) to the ecology of the entire planet (Domain IV), revealing a unified logic of self-organising, coherence-maintaining dynamics across all scales of life.",
        ),
        "test_protocols": (
            "preserve Scale-Invariant Cybernetic Principle source-accounting boundary",
        ),
        "null_results": (
            "Scale-Invariant Cybernetic Principle is not empirical validation evidence",
        ),
        "variables": ("scale_invariant_cybernetic_principle",),
        "validation_targets": ("preserve records P0R05493-P0R05496",),
        "null_controls": (
            "scale_invariant_cybernetic_principle must remain source-bounded accounting",
        ),
    },
    "scale_invariant_cybernetic_principle.citations": {
        "context_id": "citations",
        "validation_protocol": "paper0.scale_invariant_cybernetic_principle.citations",
        "canonical_statement": "The source-bounded component 'Citations:' preserves Paper 0 records P0R05497-P0R05507 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05497:citations",
            "P0R05498:citations",
            "P0R05499:citations",
            "P0R05500:citations",
            "P0R05501:citations",
            "P0R05502:citations",
            "P0R05503:citations",
            "P0R05504:citations",
            "P0R05505:citations",
            "P0R05506:citations",
            "P0R05507:citations",
        ),
        "source_formulae": (
            "P0R05497: Citations:",
            "P0R05498: Domain I - Biological Substrate",
            'P0R05499: "microtubules are proposed to sustain quantum error correction protecting coherent states at Delta 1.64 eV" -> (Hameroff & Penrose, 2014; Fisher, 2015)',
            'P0R05500: "the CISS effect in DNA geometry provides a mechanism for spin-selective information transfer" -> (Naaman & Waldeck, 2012; Michaeli et al., 2019)',
            'P0R05501: "astrocytic slow control\' stabilises neuronal networks in quasicritical regimes" -> (Fields et al., 2015; Poskanzer & Yuste, 2016)',
            'P0R05502: "cytokine signalling directly modulates qualia state-space topology" -> (Dantzer et al., 2008; Miller & Raison, 2016)',
            "P0R05503: Domain I - Expanded Biological Substrate",
            'P0R05504: "Quantum coherence in living systems is preserved through multi-scale error correction codes" -> (Huelga & Plenio, 2013; Lambert et al., 2013)',
            'P0R05505: "microtubules serve as resonant biological waveguides for consciousness-coupled fields" -> (Jibu & Yasue, 1995; Craddock et al., 2015)',
            'P0R05506: "epigenetic switching acts as a field-driven antenna system" -> (Jaenisch & Bird, 2003; Ptashne, 2007)',
            'P0R05507: "critical branching parameter sigma defines neuronal avalanche dynamics" -> (Beggs & Plenz, 2003; Shew & Plenz, 2013)',
        ),
        "test_protocols": ("preserve Citations: source-accounting boundary",),
        "null_results": ("Citations: is not empirical validation evidence",),
        "variables": ("citations",),
        "validation_targets": ("preserve records P0R05497-P0R05507",),
        "null_controls": ("citations must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class ScaleInvariantCyberneticPrincipleSpec:
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
class ScaleInvariantCyberneticPrincipleSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ScaleInvariantCyberneticPrincipleSpec, ...]
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


def build_scale_invariant_cybernetic_principle_specs(
    source_records: list[dict[str, Any]],
) -> ScaleInvariantCyberneticPrincipleSpecBundle:
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

    specs: list[ScaleInvariantCyberneticPrincipleSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ScaleInvariantCyberneticPrincipleSpec(
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
        "title": "Paper 0 " + "Scale-Invariant Cybernetic Principle" + " Specs",
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
        "next_source_boundary": "P0R05508",
    }
    return ScaleInvariantCyberneticPrincipleSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ScaleInvariantCyberneticPrincipleSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_scale_invariant_cybernetic_principle_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ScaleInvariantCyberneticPrincipleSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Scale-Invariant Cybernetic Principle" + " Specs",
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
    bundle: ScaleInvariantCyberneticPrincipleSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_scale_invariant_cybernetic_principle_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_scale_invariant_cybernetic_principle_validation_specs_{date_tag}.md"
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
