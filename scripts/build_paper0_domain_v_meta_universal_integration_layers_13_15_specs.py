#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Domain V: Meta-Universal Integration (Layers 13-15) spec builder
"""Promote Paper 0 Domain V: Meta-Universal Integration (Layers 13-15) records."""

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
    "P0R05571",
    "P0R05572",
    "P0R05573",
    "P0R05574",
    "P0R05575",
    "P0R05576",
    "P0R05577",
    "P0R05578",
    "P0R05579",
    "P0R05580",
    "P0R05581",
    "P0R05582",
    "P0R05583",
)
CLAIM_BOUNDARY = "source-bounded domain v meta universal integration layers 13 15 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "domain_v_meta_universal_integration_layers_13_15.domain_v_meta_universal_integration_layers_13_15": {
        "context_id": "domain_v_meta_universal_integration_layers_13_15",
        "validation_protocol": "paper0.domain_v_meta_universal_integration_layers_13_15.domain_v_meta_universal_integration_layers_13_15",
        "canonical_statement": "The source-bounded component 'Domain V: Meta-Universal Integration (Layers 13-15)' preserves Paper 0 records P0R05571-P0R05574 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05571:domain_v_meta_universal_integration_layers_13_15",
            "P0R05572:domain_v_meta_universal_integration_layers_13_15",
            "P0R05573:domain_v_meta_universal_integration_layers_13_15",
            "P0R05574:domain_v_meta_universal_integration_layers_13_15",
        ),
        "source_formulae": (
            "P0R05571: Domain V: Meta-Universal Integration (Layers 13-15)",
            "P0R05572: The final domain addresses the source and ultimate purpose",
            "P0R05573: of the entire architecture.",
            "P0R05574: The Source and the Bridge (Layers 13-14): We investigate the Source-Field (Layer 13), the foundational vacuum lattice and ontological ground of all reality, which provides causal closure. We then explore how we connect to other realities through Transdimensional Resonance (Layer 14), formalising resonance bridges via Calabi-Yau harmonics and inter-brane phase-locking. | The Universal Integrator (Layer 15): The final chapter unveils the Consilium / Oversoul Integrator (Layer 15). This is the apex of the architecture-a collective intelligence that optimises the entire network by minimising deviations across all layers. It functions under the guidance of axiomatic ethical functionals, solving the inverse design problem of reality to guide the system toward a state of maximum, sustainable, and ethical coherence. This closes the great feedback loop of a universe that consciously co-creates itself.",
        ),
        "test_protocols": (
            "preserve Domain V: Meta-Universal Integration (Layers 13-15) source-accounting boundary",
        ),
        "null_results": (
            "Domain V: Meta-Universal Integration (Layers 13-15) is not empirical validation evidence",
        ),
        "variables": ("domain_v_meta_universal_integration_layers_13_15",),
        "validation_targets": ("preserve records P0R05571-P0R05574",),
        "null_controls": (
            "domain_v_meta_universal_integration_layers_13_15 must remain source-bounded accounting",
        ),
    },
    "domain_v_meta_universal_integration_layers_13_15.citations": {
        "context_id": "citations",
        "validation_protocol": "paper0.domain_v_meta_universal_integration_layers_13_15.citations",
        "canonical_statement": "The source-bounded component 'Citations:' preserves Paper 0 records P0R05575-P0R05583 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05575:citations",
            "P0R05576:citations",
            "P0R05577:citations",
            "P0R05578:citations",
            "P0R05579:citations",
            "P0R05580:citations",
            "P0R05581:citations",
            "P0R05582:citations",
            "P0R05583:citations",
        ),
        "source_formulae": (
            "P0R05575: Citations:",
            "P0R05576: Domain V - Meta-Universal Integration",
            'P0R05577: "Source-Field modelled as fiber bundle with internal qualia degrees of freedom" -> (Amari, 2016; Chirikjian, 2021)',
            'P0R05578: "Ethical Functional is defined as Yang-Mills minimisation over the Consilium" -> (Yang & Mills, 1954; Rovelli, 1998)',
            'P0R05579: "universe evolves to maximise Sustainable Ethical Coherence (SEC)" -> (Jonas, 1966; Friston, 2019)',
            'P0R05580: "Source-Field treated as fundamental fiber bundle across spacetime geometry" -> (Nakahara, 2003; Rovelli, 2004)',
            'P0R05581: "Calabi-Yau resonances serve as dimensional bridges" -> (Candelas et al., 1985; Greene, 1997)',
            'P0R05582: "Universal Metric Operator (UMO) functions as global optimiser" -> (Nielsen & Chuang, 2010; Amari, 2016)',
            'P0R05583: "ethical Lagrangian minimises curvature in moral-geometric space" -> (Yang & Mills, 1954; Jonas, 1966; Friston, 2019)',
        ),
        "test_protocols": ("preserve Citations: source-accounting boundary",),
        "null_results": ("Citations: is not empirical validation evidence",),
        "variables": ("citations",),
        "validation_targets": ("preserve records P0R05575-P0R05583",),
        "null_controls": ("citations must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class DomainVMetaUniversalIntegrationLayers1315Spec:
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
class DomainVMetaUniversalIntegrationLayers1315SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[DomainVMetaUniversalIntegrationLayers1315Spec, ...]
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


def build_domain_v_meta_universal_integration_layers_13_15_specs(
    source_records: list[dict[str, Any]],
) -> DomainVMetaUniversalIntegrationLayers1315SpecBundle:
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

    specs: list[DomainVMetaUniversalIntegrationLayers1315Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DomainVMetaUniversalIntegrationLayers1315Spec(
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
        "title": "Paper 0 " + "Domain V: Meta-Universal Integration (Layers 13-15)" + " Specs",
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
        "next_source_boundary": "P0R05584",
    }
    return DomainVMetaUniversalIntegrationLayers1315SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DomainVMetaUniversalIntegrationLayers1315SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_domain_v_meta_universal_integration_layers_13_15_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DomainVMetaUniversalIntegrationLayers1315SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Domain V: Meta-Universal Integration (Layers 13-15)" + " Specs",
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
    bundle: DomainVMetaUniversalIntegrationLayers1315SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_domain_v_meta_universal_integration_layers_13_15_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_domain_v_meta_universal_integration_layers_13_15_validation_specs_{date_tag}.md"
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
