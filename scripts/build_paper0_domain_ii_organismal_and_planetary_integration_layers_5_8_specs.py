#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Domain II: Organismal and Planetary Integration (Layers 5-8) spec builder
"""Promote Paper 0 Domain II: Organismal and Planetary Integration (Layers 5-8) records."""

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
    "P0R05537",
    "P0R05538",
    "P0R05539",
    "P0R05540",
    "P0R05541",
    "P0R05542",
    "P0R05543",
    "P0R05544",
    "P0R05545",
    "P0R05546",
    "P0R05547",
    "P0R05548",
    "P0R05549",
    "P0R05550",
)
CLAIM_BOUNDARY = "source-bounded domain ii organismal and planetary integration layers 5 8 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "domain_ii_organismal_and_planetary_integration_layers_5_8.domain_ii_organismal_and_planetary_integration_layers_5_8": {
        "context_id": "domain_ii_organismal_and_planetary_integration_layers_5_8",
        "validation_protocol": "paper0.domain_ii_organismal_and_planetary_integration_layers_5_8.domain_ii_organismal_and_planetary_integration_layers_5_8",
        "canonical_statement": "The source-bounded component 'Domain II: Organismal and Planetary Integration (Layers 5-8)' preserves Paper 0 records P0R05537-P0R05540 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05537:domain_ii_organismal_and_planetary_integration_layers_5_8",
            "P0R05538:domain_ii_organismal_and_planetary_integration_layers_5_8",
            "P0R05539:domain_ii_organismal_and_planetary_integration_layers_5_8",
            "P0R05540:domain_ii_organismal_and_planetary_integration_layers_5_8",
        ),
        "source_formulae": (
            "P0R05537: Domain II: Organismal and Planetary Integration (Layers 5-8)",
            "P0R05538: This domain describes how the embodied mind integrates",
            "P0R05539: into planetary and cosmic systems.",
            "P0R05540: The Integrated Self and World (Layers 5-6): The synchronised rhythms coalesce into the unified self in the Organismal-Psychoemotional Feedback (Layer 5) layer, the seat of agency, self-awareness, and qualia. This individual field then couples with the Planetary-Biospheric (Layer 6) layer, where collective consciousness interacts with the Gaian field, exhibiting hysteresis and threshold effects. | Symbolic and Cosmic Alignment (Layers 7-8): We investigate the Geometrical-Symbolic (Layer 7) layer, where abstract symbols and language act as operators to amplify coherence. This symbolic structure helps align the organism with the Cosmic Phase-Locking (Layer 8) layer, the universal tact that synchronises life with astrophysical rhythms, guided by cosmic attractors.",
        ),
        "test_protocols": (
            "preserve Domain II: Organismal and Planetary Integration (Layers 5-8) source-accounting boundary",
        ),
        "null_results": (
            "Domain II: Organismal and Planetary Integration (Layers 5-8) is not empirical validation evidence",
        ),
        "variables": ("domain_ii_organismal_and_planetary_integration_layers_5_8",),
        "validation_targets": ("preserve records P0R05537-P0R05540",),
        "null_controls": (
            "domain_ii_organismal_and_planetary_integration_layers_5_8 must remain source-bounded accounting",
        ),
    },
    "domain_ii_organismal_and_planetary_integration_layers_5_8.citations": {
        "context_id": "citations",
        "validation_protocol": "paper0.domain_ii_organismal_and_planetary_integration_layers_5_8.citations",
        "canonical_statement": "The source-bounded component 'Citations:' preserves Paper 0 records P0R05541-P0R05550 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05541:citations",
            "P0R05542:citations",
            "P0R05543:citations",
            "P0R05544:citations",
            "P0R05545:citations",
            "P0R05546:citations",
            "P0R05547:citations",
            "P0R05548:citations",
            "P0R05549:citations",
            "P0R05550:citations",
        ),
        "source_formulae": (
            "P0R05541: Citations:",
            "P0R05542: Domain II - Organismal & Planetary",
            'P0R05543: "qualia are defined by the topological features of the consciousness manifold" -> (Giusti et al., 2015; Reimann et al., 2017)',
            'P0R05544: "the Self is modelled as a stable soliton excitation of the organismal field" -> (Friston, 2010; Friston et al., 2021)',
            'P0R05545: "collective human states are coupled to Schumann resonances" -> (Koenig, 1954; Pobachenko et al., 2006)',
            "P0R05546: Domain II - Organismal & Planetary Expansion",
            'P0R05547: "the Self emerges as a soliton stabilized by symmetry breaking" -> (Anderson, 1972; Friston, 2010)',
            'P0R05548: "qualia richness corresponds to Betti numbers of neural state manifolds" -> (Carlsson, 2009; Giusti et al., 2016)',
            'P0R05549: "biospheric memory is encoded in electromagnetic resonance layers" -> (Nickolaenko & Hayakawa, 2014; Schlegel & Fllekrug, 2020)',
            'P0R05550: "sacred geometries function as morphogenetic operators of coherence" -> (Washburn & Crowe, 1983; Gafurov et al., 2019)',
        ),
        "test_protocols": ("preserve Citations: source-accounting boundary",),
        "null_results": ("Citations: is not empirical validation evidence",),
        "variables": ("citations",),
        "validation_targets": ("preserve records P0R05541-P0R05550",),
        "null_controls": ("citations must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class DomainIiOrganismalAndPlanetaryIntegrationLayers58Spec:
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
class DomainIiOrganismalAndPlanetaryIntegrationLayers58SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[DomainIiOrganismalAndPlanetaryIntegrationLayers58Spec, ...]
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


def build_domain_ii_organismal_and_planetary_integration_layers_5_8_specs(
    source_records: list[dict[str, Any]],
) -> DomainIiOrganismalAndPlanetaryIntegrationLayers58SpecBundle:
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

    specs: list[DomainIiOrganismalAndPlanetaryIntegrationLayers58Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DomainIiOrganismalAndPlanetaryIntegrationLayers58Spec(
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
        + "Domain II: Organismal and Planetary Integration (Layers 5-8)"
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
        "next_source_boundary": "P0R05551",
    }
    return DomainIiOrganismalAndPlanetaryIntegrationLayers58SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DomainIiOrganismalAndPlanetaryIntegrationLayers58SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_domain_ii_organismal_and_planetary_integration_layers_5_8_specs(
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


def render_report(bundle: DomainIiOrganismalAndPlanetaryIntegrationLayers58SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Domain II: Organismal and Planetary Integration (Layers 5-8)" + " Specs",
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
    bundle: DomainIiOrganismalAndPlanetaryIntegrationLayers58SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_domain_ii_organismal_and_planetary_integration_layers_5_8_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_domain_ii_organismal_and_planetary_integration_layers_5_8_validation_specs_{date_tag}.md"
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
