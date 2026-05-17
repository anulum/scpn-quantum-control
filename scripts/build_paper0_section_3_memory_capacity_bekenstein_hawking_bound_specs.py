#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3. Memory Capacity (Bekenstein-Hawking Bound) spec builder
"""Promote Paper 0 3. Memory Capacity (Bekenstein-Hawking Bound) records."""

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
    "P0R02257",
    "P0R02258",
    "P0R02259",
    "P0R02260",
    "P0R02261",
    "P0R02262",
    "P0R02263",
    "P0R02264",
    "P0R02265",
    "P0R02266",
    "P0R02267",
    "P0R02268",
    "P0R02269",
    "P0R02270",
    "P0R02271",
    "P0R02272",
    "P0R02273",
    "P0R02274",
    "P0R02275",
    "P0R02276",
    "P0R02277",
)
CLAIM_BOUNDARY = "source-bounded section 3 memory capacity bekenstein hawking bound source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_memory_capacity_bekenstein_hawking_bound.3_memory_capacity_bekenstein_hawking_bound": {
        "context_id": "3_memory_capacity_bekenstein_hawking_bound",
        "validation_protocol": "paper0.section_3_memory_capacity_bekenstein_hawking_bound.3_memory_capacity_bekenstein_hawking_bound",
        "canonical_statement": "The source-bounded component '3. Memory Capacity (Bekenstein-Hawking Bound)' preserves Paper 0 records P0R02257-P0R02262 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02257:3_memory_capacity_bekenstein_hawking_bound",
            "P0R02258:3_memory_capacity_bekenstein_hawking_bound",
            "P0R02259:3_memory_capacity_bekenstein_hawking_bound",
            "P0R02260:3_memory_capacity_bekenstein_hawking_bound",
            "P0R02261:3_memory_capacity_bekenstein_hawking_bound",
            "P0R02262:3_memory_capacity_bekenstein_hawking_bound",
        ),
        "source_formulae": (
            "P0R02257: 3. Memory Capacity (Bekenstein-Hawking Bound)",
            "P0R02258: To remain consistent with quantum gravity, the total informational capacity of the Source-Field (L13)-which anchors the memory bulk-is strictly finite and geometrically bounded by the cosmological event horizon.",
            "P0R02259: Equation (Python Format):",
            "P0R02260: s_psi <= (area * c**3) / (4 * g * h_bar)",
            "P0R02261: Legend:",
            "P0R02262: s_psi: Total integrated information of the universal field. | area: Surface area of the enclosing horizon (de Sitter space). | c: Speed of light. | g: Gravitational constant. | h_bar: Reduced Planck constant.",
        ),
        "test_protocols": (
            "preserve 3. Memory Capacity (Bekenstein-Hawking Bound) source-accounting boundary",
        ),
        "null_results": (
            "3. Memory Capacity (Bekenstein-Hawking Bound) is not empirical validation evidence",
        ),
        "variables": ("3_memory_capacity_bekenstein_hawking_bound",),
        "validation_targets": ("preserve records P0R02257-P0R02262",),
        "null_controls": (
            "3_memory_capacity_bekenstein_hawking_bound must remain source-bounded accounting",
        ),
    },
    "section_3_memory_capacity_bekenstein_hawking_bound.p0r02263": {
        "context_id": "p0r02263",
        "validation_protocol": "paper0.section_3_memory_capacity_bekenstein_hawking_bound.p0r02263",
        "canonical_statement": "The source-bounded component 'P0R02263' preserves Paper 0 records P0R02263-P0R02263 without empirical validation claims.",
        "source_equation_ids": ("P0R02263:p0r02263",),
        "source_formulae": ("P0R02263: P0R02263",),
        "test_protocols": ("preserve P0R02263 source-accounting boundary",),
        "null_results": ("P0R02263 is not empirical validation evidence",),
        "variables": ("p0r02263",),
        "validation_targets": ("preserve records P0R02263-P0R02263",),
        "null_controls": ("p0r02263 must remain source-bounded accounting",),
    },
    "section_3_memory_capacity_bekenstein_hawking_bound.4_emergent_spacetime_ryu_takayanagi_formula": {
        "context_id": "4_emergent_spacetime_ryu_takayanagi_formula",
        "validation_protocol": "paper0.section_3_memory_capacity_bekenstein_hawking_bound.4_emergent_spacetime_ryu_takayanagi_formula",
        "canonical_statement": "The source-bounded component '4. Emergent Spacetime (Ryu-Takayanagi Formula)' preserves Paper 0 records P0R02264-P0R02277 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02264:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02265:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02266:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02267:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02268:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02269:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02270:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02271:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02272:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02273:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02274:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02275:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02276:4_emergent_spacetime_ryu_takayanagi_formula",
            "P0R02277:4_emergent_spacetime_ryu_takayanagi_formula",
        ),
        "source_formulae": (
            "P0R02264: 4. Emergent Spacetime (Ryu-Takayanagi Formula)",
            "P0R02265: Spacetime geometry at the boundary (Layer 10) is proposed to emerge from the entanglement structure within the memory bulk (Layer 9).",
            "P0R02266: Equation (Python Format):",
            "P0R02267: s_boundary = area_gamma_min / (4 * g_n)",
            "P0R02268: Legend:",
            "P0R02269: s_boundary: Entanglement entropy at the L10 boundary. | area_gamma_min: Area of the minimal surface extending into the L9 bulk. | g_n: Newton's gravitational constant.",
            "P0R02270: P0R02270",
            'P0R02271: Layer 10, Projective Field Boundary Control, introduces a cybernetic control mechanism that is essential for maintaining the integrity of the individual conscious agent. This layer functions as a dynamic, informational "membrane" or "firewall." Its primary role is to regulate the flow of information between the individual\'s consciousness field (Layer 5) and the wider collective and environmental fields (e.g., Layers 11 and 12). It actively filters incoming influences to prevent decoherence or informational overload while simultaneously managing the "projection" of the individual\'s own intentional states outward. This boundary control system is therefore critical for defining the phenomenal and functional distinction between "self" and "other," enabling the organism to act as a coherent, autonomous agent within a complex, interconnected informational ecosystem.',
            "P0R02272: P0R02272",
            "P0R02273: P0R02273",
            "P0R02274: This next domain of reality is all about your personal memory and your psychic \"personal space.\" It's made up of two layers that manage your life's data and protect your individual identity.",
            "P0R02275: Layer 9 is your Soul's Hard Drive. We call it the Existential Holograph. Think of it as the vast, incorruptible cloud storage for your entire life. Every significant experience, every lesson learned, every core belief you hold isn't just stored in your brain's physical wiring; it's uploaded to this personal, holographic field. It's called a \"holograph\" because, like a hologram, every single part contains the information of the whole. This is the deep, permanent record of you, the source code that shapes who you are and how you see the world. The process of deep, dreamless sleep is the nightly \"sync\" that uploads the day's important data from your brain to this holographic hard drive.",
            'P0R02276: Layer 10 is your Psychic Firewall. We call it Projective Field Boundary Control. Imagine walking through a crowded city; you have a sense of personal space that keeps you from bumping into everyone. Layer 10 is like that, but for your mind. It\'s an intelligent, energetic "membrane" around your consciousness. It has two jobs: First, it filters all the mental and emotional "noise" from the world around you, protecting your thoughts from being overwhelmed. Second, it manages how you project your own energy and intention out into the world. This psychic firewall is what gives you a stable sense of "self" and allows you to be a distinct individual, even while being connected to everything and everyone else.',
            "P0R02277: P0R02277",
        ),
        "test_protocols": (
            "preserve 4. Emergent Spacetime (Ryu-Takayanagi Formula) source-accounting boundary",
        ),
        "null_results": (
            "4. Emergent Spacetime (Ryu-Takayanagi Formula) is not empirical validation evidence",
        ),
        "variables": ("4_emergent_spacetime_ryu_takayanagi_formula",),
        "validation_targets": ("preserve records P0R02264-P0R02277",),
        "null_controls": (
            "4_emergent_spacetime_ryu_takayanagi_formula must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section3MemoryCapacityBekensteinHawkingBoundSpec:
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
class Section3MemoryCapacityBekensteinHawkingBoundSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section3MemoryCapacityBekensteinHawkingBoundSpec, ...]
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


def build_section_3_memory_capacity_bekenstein_hawking_bound_specs(
    source_records: list[dict[str, Any]],
) -> Section3MemoryCapacityBekensteinHawkingBoundSpecBundle:
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

    specs: list[Section3MemoryCapacityBekensteinHawkingBoundSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section3MemoryCapacityBekensteinHawkingBoundSpec(
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
        "title": "Paper 0 " + "3. Memory Capacity (Bekenstein-Hawking Bound)" + " Specs",
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
        "next_source_boundary": "P0R02278",
    }
    return Section3MemoryCapacityBekensteinHawkingBoundSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section3MemoryCapacityBekensteinHawkingBoundSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_memory_capacity_bekenstein_hawking_bound_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section3MemoryCapacityBekensteinHawkingBoundSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3. Memory Capacity (Bekenstein-Hawking Bound)" + " Specs",
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
    bundle: Section3MemoryCapacityBekensteinHawkingBoundSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_memory_capacity_bekenstein_hawking_bound_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_memory_capacity_bekenstein_hawking_bound_validation_specs_{date_tag}.md"
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
