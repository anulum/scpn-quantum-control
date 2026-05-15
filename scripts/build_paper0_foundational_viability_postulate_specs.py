#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 foundational viability postulate spec builder
"""Promote Paper 0 foundational viability and Psi-field postulate records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(464, 506))
CLAIM_BOUNDARY = "source-bounded foundational viability postulate; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
COUPLING_EQUATION = "H_int = -lambda * Psi_s * sigma"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "foundational_viability_postulate.three_pillar_viability_frame": {
        "context_id": "three_pillar_viability_frame",
        "validation_protocol": "paper0.foundational_viability_postulate.three_pillar_viability_frame",
        "canonical_statement": (
            "Section 1.2 frames viability through three pillars: ontological postulate, "
            "derived physical interactions, and multiscale architecture."
        ),
        "source_equation_ids": (
            "P0R00464:section_1_2_foundational_viability",
            "P0R00465:three_core_pillars",
            "P0R00468:hierarchy_rg_bidirectional_causality",
        ),
        "source_formulae": (
            "Foundational Viability: Internal Consistency & Postulates",
            "ontological postulate",
            "derived physical interactions",
            "multiscale architecture",
            "Renormalisation Group flows",
        ),
        "test_protocols": ("classify three viability pillars and reject evidence promotion",),
        "null_results": ("internal consistency framing is not empirical validation",),
        "variables": (
            "ontological_postulate",
            "derived_interactions",
            "multiscale_architecture",
        ),
        "validation_targets": (
            "preserve Psi-field primitive-ontology pillar",
            "preserve U(1)/FIM interaction-derivation pillar",
            "preserve hierarchy/RG bidirectional-causality pillar",
        ),
        "null_controls": (
            "internal-consistency-as-empirical-validation control must be rejected",
            "unmapped-viability-pillar control must be rejected",
        ),
    },
    "foundational_viability_postulate.dynamic_spine_viability": {
        "context_id": "dynamic_spine_viability",
        "validation_protocol": "paper0.foundational_viability_postulate.dynamic_spine_viability",
        "canonical_statement": (
            "The dynamic spine is framed as UPDE generalising Kuramoto with an "
            "information-geometric/FIM lift, quasicriticality, and MS-QEC protection."
        ),
        "source_equation_ids": (
            "P0R00467:upde_kuramoto_fim_spine",
            "P0R00476:upde_master_equation",
            "P0R00477:quasicriticality",
            "P0R00478:ms_qec_protection",
        ),
        "source_formulae": (
            "UPDE multi-scale generalisation of the Kuramoto model",
            "information-geometric lift",
            "FIM-based",
            "Quasicriticality",
            "MS-QEC",
        ),
        "test_protocols": ("preserve dynamic-spine labels without treating them as data",),
        "null_results": ("dynamic-spine overview is not numerical validation evidence",),
        "variables": ("upde", "kuramoto", "fim", "quasicriticality", "ms_qec"),
        "validation_targets": (
            "preserve UPDE/Kuramoto relation",
            "preserve information-geometric FIM bridge",
            "preserve quasicriticality and MS-QEC as later validation targets",
        ),
        "null_controls": (
            "overview-as-upde-validation control must be rejected",
            "unverified-ms-qec-shield control must be rejected",
        ),
    },
    "foundational_viability_postulate.hint_component_viability": {
        "context_id": "hint_component_viability",
        "validation_protocol": "paper0.foundational_viability_postulate.hint_component_viability",
        "canonical_statement": (
            "The integration text frames H_int by requiring viability of Psi_s, "
            "interaction derivation, sigma_info coupling, and sigma_phys persistence."
        ),
        "source_equation_ids": (
            "P0R00487:H_int=-lambda*Psi_s*sigma",
            "P0R00489:u1_derivation_boundary",
            "P0R00490:sigma_info_coupling",
            "P0R00491:sigma_phys_persistence",
        ),
        "source_formulae": (
            COUPLING_EQUATION,
            "U(1) gauge symmetry",
            "sigma_info",
            "sigma_phys",
            "stable, coherent, and protected collective state variable",
        ),
        "test_protocols": ("preserve H_int component viability boundaries",),
        "null_results": ("H_int component map is not a completed derivation or experiment",),
        "variables": ("Psi_s", "lambda", "sigma_info", "sigma_phys", "H_int"),
        "validation_targets": (
            "preserve Psi_s viability requirement",
            "preserve U(1) derivation boundary",
            "preserve sigma_info and sigma_phys distinction",
        ),
        "null_controls": (
            "component-map-as-derivation control must be rejected",
            "unmeasured-sigma-control must be rejected",
        ),
    },
    "foundational_viability_postulate.psi_primitive_ontology_boundary": {
        "context_id": "psi_primitive_ontology_boundary",
        "validation_protocol": "paper0.foundational_viability_postulate.psi_primitive_ontology_boundary",
        "canonical_statement": (
            "The Psi-field is framed as an irreducible ontological primitive and "
            "Hierarchical Field Monism, but explicitly as a generative hypothesis."
        ),
        "source_equation_ids": (
            "P0R00466:primitive_ontology_hfm",
            "P0R00496:irreducible_ontological_primitive",
            "P0R00497:generative_hypothesis_boundary",
        ),
        "source_formulae": (
            "primitive ontology",
            "Hierarchical Field Monism",
            "irreducible ontological primitive",
            "not as a dogmatic assertion",
            "generative hypothesis",
        ),
        "test_protocols": (
            "preserve primitive-ontology status and generative-hypothesis boundary",
        ),
        "null_results": ("ontological postulate is not empirical detection of Psi-field quanta",),
        "variables": ("psi_field", "primitive_ontology", "hfm", "generative_hypothesis"),
        "validation_targets": (
            "preserve HFM label",
            "preserve non-dogmatic generative-hypothesis boundary",
            "reject primitive-ontology-as-detection promotion",
        ),
        "null_controls": (
            "ontology-as-detection control must be rejected",
            "dogma-boundary-removal control must be rejected",
        ),
    },
    "foundational_viability_postulate.psi_complex_scalar_field": {
        "context_id": "psi_complex_scalar_field",
        "validation_protocol": "paper0.foundational_viability_postulate.psi_complex_scalar_field",
        "canonical_statement": (
            "The Psi-field postulate is formalised as a complex scalar field with "
            "spin-0 bosonic quanta, global U(1) phase symmetry, and magnitude/phase decomposition."
        ),
        "source_equation_ids": (
            "P0R00500:psi_complex_scalar_field",
            "P0R00502:spin0_u1_phase_symmetry",
            "P0R00503:psi_magnitude_phase_decomposition",
        ),
        "source_formulae": (
            "complex scalar field",
            "spin-0 bosons",
            "global U(1) phase symmetry",
            "Psi = |Psi| e^{i theta}",
            "U(1) gauge + FIM",
        ),
        "test_protocols": ("preserve Psi-field quantum-number and decomposition bookkeeping",),
        "null_results": ("formal field-theory analogy is not direct experimental detection",),
        "variables": ("spin", "statistics", "symmetry", "decomposition"),
        "validation_targets": (
            "preserve spin-0 scalar-field assignment",
            "preserve global U(1) phase-symmetry assignment",
            "preserve magnitude/phase decomposition",
        ),
        "null_controls": (
            "complex-scalar-formalisation-as-detection control must be rejected",
            "missing-u1-symmetry control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class FoundationalViabilityPostulateSpec:
    """Foundational viability postulate spec promoted from Paper 0 records."""

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
class FoundationalViabilityPostulateSpecBundle:
    """Foundational viability postulate specs plus source coverage summary."""

    specs: tuple[FoundationalViabilityPostulateSpec, ...]
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


def build_foundational_viability_postulate_specs(
    source_records: list[dict[str, Any]],
) -> FoundationalViabilityPostulateSpecBundle:
    """Build source-covered foundational viability postulate specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[FoundationalViabilityPostulateSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            FoundationalViabilityPostulateSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_source_fixture",
                domain_review_status="requires_domain_review_before_scientific_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Foundational Viability Postulate Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "pillar_count": 3,
        "physics_postulate_count": 4,
        "next_source_boundary": "P0R00506",
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [],
    }
    return FoundationalViabilityPostulateSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> FoundationalViabilityPostulateSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = [
        record
        for record in load_jsonl(ledger_path)
        if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS
    ]
    return build_foundational_viability_postulate_specs(records)


def write_outputs(
    bundle: FoundationalViabilityPostulateSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_foundational_viability_postulate_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_foundational_viability_postulate_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: FoundationalViabilityPostulateSpecBundle) -> str:
    """Render a compact Markdown report for promoted postulate specs."""
    lines = [
        "# Paper 0 Foundational Viability Postulate Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Specs: {bundle.summary['spec_count']}",
        f"- Pillars: {bundle.summary['pillar_count']}",
        f"- Physical postulate components: {bundle.summary['physics_postulate_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"- `{spec.key}`",
                f"  - Context: `{spec.context_id}`",
                f"  - Statement: {spec.canonical_statement}",
                f"  - Formulae: {', '.join(spec.source_formulae)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build foundational viability postulate specs and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
