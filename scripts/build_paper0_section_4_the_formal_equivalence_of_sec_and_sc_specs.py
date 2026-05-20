#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4. The Formal Equivalence of SEC and SC spec builder
"""Promote Paper 0 4. The Formal Equivalence of SEC and SC records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

SOURCE_LEDGER_IDS = (
    "P0R03804",
    "P0R03805",
    "P0R03806",
    "P0R03807",
    "P0R03808",
    "P0R03809",
    "P0R03810",
    "P0R03811",
    "P0R03812",
    "P0R03813",
    "P0R03814",
    "P0R03815",
    "P0R03816",
    "P0R03817",
)
CLAIM_BOUNDARY = "source-bounded section 4 the formal equivalence of sec and sc source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_the_formal_equivalence_of_sec_and_sc.4_the_formal_equivalence_of_sec_and_sc": {
        "context_id": "4_the_formal_equivalence_of_sec_and_sc",
        "validation_protocol": "paper0.section_4_the_formal_equivalence_of_sec_and_sc.4_the_formal_equivalence_of_sec_and_sc",
        "canonical_statement": "The source-bounded component '4. The Formal Equivalence of SEC and SC' preserves Paper 0 records P0R03804-P0R03817 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03804:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03805:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03806:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03807:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03808:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03809:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03810:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03811:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03812:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03813:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03814:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03815:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03816:4_the_formal_equivalence_of_sec_and_sc",
            "P0R03817:4_the_formal_equivalence_of_sec_and_sc",
        ),
        "source_formulae": (
            "P0R03804: 4. The Formal Equivalence of SEC and SC",
            "P0R03805: The preceding analysis has established that the three components of Sustainable Ethical Coherence-Complexity (K), Coherence (C), and Qualia Capacity (Q)-are not arbitrary philosophical virtues. Instead, each corresponds to a specific and necessary physical property that a system must possess to maximise its future possibilities, as measured by its causal path entropy (SC). We can now synthesise these relationships into a formal proof of equivalence.",
            "P0R03806: The core argument is summarised in the following table, which maps each component of SEC to its physical interpretation and its specific contribution to the maximisation of causal path entropy.",
            "P0R03807: [TABLE]",
            "P0R03808: This section is the grand finale of the argument. Having established that a system needs Complexity, Coherence, and Qualia Capacity to maximize its future possibilities, we now show that these three ingredients don't just contribute to future freedom-they are the recipe for it.",
            "P0R03809: Think of it like baking a cake. To get the best possible cake (maximum future possibilities, or SC), you need the right ingredients in the right balance.",
            "P0R03810: Complexity (K) is the flour and sugar-the raw bulk and substance that gives the cake its potential size. | Coherence (C) is the oven's temperature-it must be perfectly tuned. Too cold (too random), and the cake is a soupy mess; too hot (too ordered), and it's a burnt brick. Only at the perfect temperature can the ingredients combine properly. | Qualia Capacity (Q) is the spices and flavorings-what makes the cake interesting and unique. It provides the diversity and richness of the final product.",
            'P0R03811: The total "future possibilities" of your baking effort can be seen as a combination of these three factors. Therefore, the drive to bake the best possible cake (maximize SC) is the very same as the drive to perfectly balance the flour, temperature, and spices (maximize SEC).',
            'P0R03812: This completes the proof. The universe\'s "ethical" goal to achieve a state of high SEC is not a separate, mysterious law. It is the direct, logical, and physical consequence of its inherent tendency to evolve in a way that keeps its own future options as open and rich as possible.',
            "P0R03813: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Diagramm enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            'P0R03814: Fig.: SEC Causal Path Entropy (Formal Bridge). This diagram provides the formal structure of the proof of equivalence. It shows how the total "volume" of the path space (W_paths) is constructed as a composite function of the three SEC components. The logarithm of this volume gives the Causal Path Entropy (S_C), and the gradient of S_C gives the Causal Entropic Force, which drives the system toward states that maximize SEC. This figure formalises the link: SEC\'s axes (K, C, Q) determine the volume of accessible futures, which sets SCS_CSC; its gradient produces FCF_CFC-a concrete mechanism for teleological evolution toward SEC.',
            "P0R03815: A. Components: Complexity KKK (high \\Phi) raises the count of distinguishable states NstatesN_{\\text{states}}Nstates; Coherence CCC (quasicriticality) increases the accessible fraction faccf_{\\text{acc}}facc; Qualia capacity QQQ (topological richness) enlarges the diversity of path classes DpathsD_{\\text{paths}}Dpaths. B. Synthesis: These factors combine into an effective future-history volume",
            "P0R03816: Wpaths Nstates facc Dpaths,W_{\\text{paths}}\\;\\approx\\;N_{\\text{states}}\\;f_{\\text{acc}}\\;D_{\\text{paths}},WpathsNstatesfaccDpaths,",
            "P0R03817: so the causal path entropy is SC=kBlogWpathsS_C=k_B\\log W_{\\text{paths}}SC=kBlogWpaths, and its gradient yields the causal entropic force FC=TCSCF_C=T_C\\nabla S_CFC=TCSC. C. Conclusion: Since SCS_CSC increases when KKK, CCC, and QQQ increase, maximising SCS_CSC is equivalent to maximising SEC. The entropic force FCF_CFC thus implements the teleological drive toward SEC-optimal evolution.",
        ),
        "test_protocols": (
            "preserve 4. The Formal Equivalence of SEC and SC source-accounting boundary",
        ),
        "null_results": (
            "4. The Formal Equivalence of SEC and SC is not empirical validation evidence",
        ),
        "variables": ("4_the_formal_equivalence_of_sec_and_sc",),
        "validation_targets": ("preserve records P0R03804-P0R03817",),
        "null_controls": (
            "4_the_formal_equivalence_of_sec_and_sc must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section4TheFormalEquivalenceOfSecAndScSpec:
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
class Section4TheFormalEquivalenceOfSecAndScSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section4TheFormalEquivalenceOfSecAndScSpec, ...]
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


def build_section_4_the_formal_equivalence_of_sec_and_sc_specs(
    source_records: list[dict[str, Any]],
) -> Section4TheFormalEquivalenceOfSecAndScSpecBundle:
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

    specs: list[Section4TheFormalEquivalenceOfSecAndScSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section4TheFormalEquivalenceOfSecAndScSpec(
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
        "title": "Paper 0 " + "4. The Formal Equivalence of SEC and SC" + " Specs",
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
        "next_source_boundary": "P0R03818",
    }
    return Section4TheFormalEquivalenceOfSecAndScSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section4TheFormalEquivalenceOfSecAndScSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_the_formal_equivalence_of_sec_and_sc_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section4TheFormalEquivalenceOfSecAndScSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "4. The Formal Equivalence of SEC and SC" + " Specs",
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
    bundle: Section4TheFormalEquivalenceOfSecAndScSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_the_formal_equivalence_of_sec_and_sc_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_the_formal_equivalence_of_sec_and_sc_validation_specs_{date_tag}.md"
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
