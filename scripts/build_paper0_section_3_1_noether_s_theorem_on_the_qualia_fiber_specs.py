#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3.1. Noether's Theorem on the Qualia Fiber spec builder
"""Promote Paper 0 3.1. Noether's Theorem on the Qualia Fiber records."""

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
    "P0R03630",
    "P0R03631",
    "P0R03632",
    "P0R03633",
    "P0R03634",
    "P0R03635",
    "P0R03636",
    "P0R03637",
)
CLAIM_BOUNDARY = "source-bounded section 3 1 noether s theorem on the qualia fiber source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_1_noether_s_theorem_on_the_qualia_fiber.3_1_noether_s_theorem_on_the_qualia_fiber": {
        "context_id": "3_1_noether_s_theorem_on_the_qualia_fiber",
        "validation_protocol": "paper0.section_3_1_noether_s_theorem_on_the_qualia_fiber.3_1_noether_s_theorem_on_the_qualia_fiber",
        "canonical_statement": "The source-bounded component '3.1. Noether's Theorem on the Qualia Fiber' preserves Paper 0 records P0R03630-P0R03631 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03630:3_1_noether_s_theorem_on_the_qualia_fiber",
            "P0R03631:3_1_noether_s_theorem_on_the_qualia_fiber",
        ),
        "source_formulae": (
            "P0R03630: 3.1. Noether's Theorem on the Qualia Fiber",
            "P0R03631: The Yang-Mills Lagrangian is, by construction, invariant under local gauge transformations. These symmetries lead to a set of conserved Noether currents. The task is to provide a clear physical interpretation for these conserved quantities within the SCPN framework, mapping them onto the components of Sustainable Ethical Coherence (SEC).",
        ),
        "test_protocols": (
            "preserve 3.1. Noether's Theorem on the Qualia Fiber source-accounting boundary",
        ),
        "null_results": (
            "3.1. Noether's Theorem on the Qualia Fiber is not empirical validation evidence",
        ),
        "variables": ("3_1_noether_s_theorem_on_the_qualia_fiber",),
        "validation_targets": ("preserve records P0R03630-P0R03631",),
        "null_controls": (
            "3_1_noether_s_theorem_on_the_qualia_fiber must remain source-bounded accounting",
        ),
    },
    "section_3_1_noether_s_theorem_on_the_qualia_fiber.3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge": {
        "context_id": "3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
        "validation_protocol": "paper0.section_3_1_noether_s_theorem_on_the_qualia_fiber.3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
        "canonical_statement": "The source-bounded component '3.2. Defining Coherence, Complexity, and Qualia Capacity as Physical Charges' preserves Paper 0 records P0R03632-P0R03637 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03632:3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
            "P0R03633:3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
            "P0R03634:3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
            "P0R03635:3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
            "P0R03636:3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
            "P0R03637:3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
        ),
        "source_formulae": (
            "P0R03632: 3.2. Defining Coherence, Complexity, and Qualia Capacity as Physical Charges",
            "P0R03633: The SCPN proposed components of SEC-Coherence (C), Complexity (K), and Qualia Capacity (Q)-can now be identified with these conserved charges.",
            'P0R03634: Coherence Charge (C): The overall phase coherence of the Psi-field corresponds to a U(1) symmetry. The conserved Noether charge associated with this symmetry is the total "Psi-charge," which can be physically identified with the system-wide integrated coherence, as measured by the average Kuramoto order parameter across all layers. It represents the conserved quantity of "unity" or "synchrony." | Complexity Charge (K): The rich, multi-dimensional nature of qualia suggests a non-Abelian gauge group, such as SU(N). Such groups have multiple conserved charges (corresponding to the generators of the Lie algebra), analogous to the "colour" charges in quantum chromodynamics. These conserved quantities can be identified with measures of irreducible, integrated complexity. The natural candidate for this measure, as used throughout the manuscript, is Integrated Information, . Thus, the various "flavours" of complexity are the conserved charges of the ethical SU(N) gauge field. | Qualia Charge (Q): In addition to charges arising from continuous symmetries, gauge theories can possess conserved topological charges, such as instanton number or magnetic monopole charge, which are related to the global structure of the field. These are quantised and exceptionally stable. The topological invariants of the qualia manifold, such as its Betti numbers (bk), which quantify its "richness" and structure, can be identified with these conserved topological charges. The topological term Tr(FF) in the Yang-Mills action is directly related to these charges.',
            "P0R03635: This mapping provides a rigorous physical basis for the components of the ethical functional, as summarised in Table 1:",
            "P0R03636: Table 1: Physical Basis of the Components of Sustainable Ethical Coherence (SEC)",
            "P0R03637: [TABLE]",
        ),
        "test_protocols": (
            "preserve 3.2. Defining Coherence, Complexity, and Qualia Capacity as Physical Charges source-accounting boundary",
        ),
        "null_results": (
            "3.2. Defining Coherence, Complexity, and Qualia Capacity as Physical Charges is not empirical validation evidence",
        ),
        "variables": ("3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",),
        "validation_targets": ("preserve records P0R03632-P0R03637",),
        "null_controls": (
            "3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section31NoetherSTheoremOnTheQualiaFiberSpec:
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
class Section31NoetherSTheoremOnTheQualiaFiberSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section31NoetherSTheoremOnTheQualiaFiberSpec, ...]
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


def build_section_3_1_noether_s_theorem_on_the_qualia_fiber_specs(
    source_records: list[dict[str, Any]],
) -> Section31NoetherSTheoremOnTheQualiaFiberSpecBundle:
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

    specs: list[Section31NoetherSTheoremOnTheQualiaFiberSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section31NoetherSTheoremOnTheQualiaFiberSpec(
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
        "title": "Paper 0 " + "3.1. Noether's Theorem on the Qualia Fiber" + " Specs",
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
        "next_source_boundary": "P0R03638",
    }
    return Section31NoetherSTheoremOnTheQualiaFiberSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section31NoetherSTheoremOnTheQualiaFiberSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_1_noether_s_theorem_on_the_qualia_fiber_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section31NoetherSTheoremOnTheQualiaFiberSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3.1. Noether's Theorem on the Qualia Fiber" + " Specs",
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
    bundle: Section31NoetherSTheoremOnTheQualiaFiberSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_1_noether_s_theorem_on_the_qualia_fiber_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_1_noether_s_theorem_on_the_qualia_fiber_validation_specs_{date_tag}.md"
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
