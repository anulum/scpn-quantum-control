#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 VI. The Unified Geometric Principle (UGP) spec builder
"""Promote Paper 0 VI. The Unified Geometric Principle (UGP) records."""

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
    "P0R06039",
    "P0R06040",
    "P0R06041",
    "P0R06042",
    "P0R06043",
    "P0R06044",
    "P0R06045",
    "P0R06046",
)
CLAIM_BOUNDARY = "source-bounded vi the unified geometric principle ugp source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "vi_the_unified_geometric_principle_ugp.vi_the_unified_geometric_principle_ugp": {
        "context_id": "vi_the_unified_geometric_principle_ugp",
        "validation_protocol": "paper0.vi_the_unified_geometric_principle_ugp.vi_the_unified_geometric_principle_ugp",
        "canonical_statement": "The source-bounded component 'VI. The Unified Geometric Principle (UGP)' preserves Paper 0 records P0R06039-P0R06040 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06039:vi_the_unified_geometric_principle_ugp",
            "P0R06040:vi_the_unified_geometric_principle_ugp",
        ),
        "source_formulae": (
            "P0R06039: VI. The Unified Geometric Principle (UGP)",
            "P0R06040: Geometry is the universal language of organisation (Axiom 2). The UGP asserts that InformationGeometry. This principle manifests across all scales: from the topology of QEC codes (L1) and DNA chirality (L3), to the geometry of the Qualia Manifold (L5), the action of Symbols as Rotors (L7), the hyperbolic geometry of MERA networks (L9), and the potential landscape defined by the Ethical Functional (L15).",
        ),
        "test_protocols": (
            "preserve VI. The Unified Geometric Principle (UGP) source-accounting boundary",
        ),
        "null_results": (
            "VI. The Unified Geometric Principle (UGP) is not empirical validation evidence",
        ),
        "variables": ("vi_the_unified_geometric_principle_ugp",),
        "validation_targets": ("preserve records P0R06039-P0R06040",),
        "null_controls": (
            "vi_the_unified_geometric_principle_ugp must remain source-bounded accounting",
        ),
    },
    "vi_the_unified_geometric_principle_ugp.vii_symmetry_principles_preservation_and_breaking": {
        "context_id": "vii_symmetry_principles_preservation_and_breaking",
        "validation_protocol": "paper0.vi_the_unified_geometric_principle_ugp.vii_symmetry_principles_preservation_and_breaking",
        "canonical_statement": "The source-bounded component 'VII. Symmetry Principles: Preservation and Breaking' preserves Paper 0 records P0R06041-P0R06043 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06041:vii_symmetry_principles_preservation_and_breaking",
            "P0R06042:vii_symmetry_principles_preservation_and_breaking",
            "P0R06043:vii_symmetry_principles_preservation_and_breaking",
        ),
        "source_formulae": (
            "P0R06041: VII. Symmetry Principles: Preservation and Breaking",
            "P0R06042: Symmetry governs the SCPN.",
            "P0R06043: Symmetry Preservation (Noether's Theorem): Symmetries lead to conserved quantities (e.g., Gauge Symmetry Conservation of Coherence). | Spontaneous Symmetry Breaking (SSB): Essential for the emergence of structure, such as the formation of the Self (L5 VEV) and the differentiation of the Source-Field (L13).",
        ),
        "test_protocols": (
            "preserve VII. Symmetry Principles: Preservation and Breaking source-accounting boundary",
        ),
        "null_results": (
            "VII. Symmetry Principles: Preservation and Breaking is not empirical validation evidence",
        ),
        "variables": ("vii_symmetry_principles_preservation_and_breaking",),
        "validation_targets": ("preserve records P0R06041-P0R06043",),
        "null_controls": (
            "vii_symmetry_principles_preservation_and_breaking must remain source-bounded accounting",
        ),
    },
    "vi_the_unified_geometric_principle_ugp.viii_energetics_and_metabolism_of_the_scpn": {
        "context_id": "viii_energetics_and_metabolism_of_the_scpn",
        "validation_protocol": "paper0.vi_the_unified_geometric_principle_ugp.viii_energetics_and_metabolism_of_the_scpn",
        "canonical_statement": "The source-bounded component 'VIII. Energetics and Metabolism of the SCPN' preserves Paper 0 records P0R06044-P0R06046 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06044:viii_energetics_and_metabolism_of_the_scpn",
            "P0R06045:viii_energetics_and_metabolism_of_the_scpn",
            "P0R06046:viii_energetics_and_metabolism_of_the_scpn",
        ),
        "source_formulae": (
            "P0R06044: VIII. Energetics and Metabolism of the SCPN",
            "P0R06045: The SCPN operates far from equilibrium as a Dissipative Structure.",
            "P0R06046: Generalised Free Energy (GFE): The system minimises GFE, integrating thermodynamic and informational components: FGFE=FThermoTEffSInfo. | The Cost of Consciousness: We posit a scaling relationship: RMetabolic (>1). | Psi-Field Efficiency: The Psi-field enhances metabolic efficiency by increasing quantum coherence (Negentropy Injection).",
        ),
        "test_protocols": (
            "preserve VIII. Energetics and Metabolism of the SCPN source-accounting boundary",
        ),
        "null_results": (
            "VIII. Energetics and Metabolism of the SCPN is not empirical validation evidence",
        ),
        "variables": ("viii_energetics_and_metabolism_of_the_scpn",),
        "validation_targets": ("preserve records P0R06044-P0R06046",),
        "null_controls": (
            "viii_energetics_and_metabolism_of_the_scpn must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ViTheUnifiedGeometricPrincipleUgpSpec:
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
class ViTheUnifiedGeometricPrincipleUgpSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ViTheUnifiedGeometricPrincipleUgpSpec, ...]
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


def build_vi_the_unified_geometric_principle_ugp_specs(
    source_records: list[dict[str, Any]],
) -> ViTheUnifiedGeometricPrincipleUgpSpecBundle:
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

    specs: list[ViTheUnifiedGeometricPrincipleUgpSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ViTheUnifiedGeometricPrincipleUgpSpec(
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
        "title": "Paper 0 " + "VI. The Unified Geometric Principle (UGP)" + " Specs",
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
        "next_source_boundary": "P0R06047",
    }
    return ViTheUnifiedGeometricPrincipleUgpSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ViTheUnifiedGeometricPrincipleUgpSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_vi_the_unified_geometric_principle_ugp_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ViTheUnifiedGeometricPrincipleUgpSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "VI. The Unified Geometric Principle (UGP)" + " Specs",
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
    bundle: ViTheUnifiedGeometricPrincipleUgpSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_vi_the_unified_geometric_principle_ugp_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_vi_the_unified_geometric_principle_ugp_validation_specs_{date_tag}.md"
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
