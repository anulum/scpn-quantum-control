#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom I Psi-field spec builder
"""Promote Paper 0 Axiom I Psi-field records into source-bounded specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(670, 703))
BLANK_SEPARATOR_IDS = ("P0R00685", "P0R00702")
CLAIM_BOUNDARY = "source-bounded Axiom I Psi-field map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_i_psi_field.axiom_i_source_boundary": {
        "context_id": "axiom_i_source_boundary",
        "validation_protocol": "paper0.axiom_i_psi_field.axiom_i_source_boundary",
        "canonical_statement": (
            "Axiom I is promoted only as a metaphysical source-boundary claim: "
            "consciousness is treated as an irreducible ontological primitive."
        ),
        "source_equation_ids": (
            "P0R00670:axiom_i_header",
            "P0R00674:foundational_axiom_header",
            "P0R00679:axiom_i_foundational_postulate",
            "P0R00700:axiom_i_ontological_primitive",
        ),
        "source_formulae": (
            "Axiom I: The Primacy of Consciousness",
            "consciousness as irreducible ontological primitive",
            "Axiom of Existence",
            "logical starting point",
        ),
        "test_protocols": ("classify Axiom I source role",),
        "null_results": ("Axiom I is not an empirical result",),
        "variables": ("Psi", "consciousness", "ontology"),
        "validation_targets": (
            "preserve ontological-primitive role",
            "preserve axiom-not-observation boundary",
            "preserve next-model-class boundary",
        ),
        "null_controls": (
            "axiom-as-measurement control must be rejected",
            "missing-axiom-boundary control must be rejected",
        ),
    },
    "axiom_i_psi_field.psi_field_formalisation": {
        "context_id": "psi_field_formalisation",
        "validation_protocol": "paper0.axiom_i_psi_field.psi_field_formalisation",
        "canonical_statement": (
            "The source formalises fundamental consciousness as a universal complex "
            "scalar Psi-field and bounds matter as emergent modes or excitations."
        ),
        "source_equation_ids": (
            "P0R00680:psi_field_universal_complex_scalar",
            "P0R00683:psi_field_scientific_language",
            "P0R00684:psi_field_ocean_analogy",
            "P0R00701:psi_field_not_emergent_matter_modalities",
        ),
        "source_formulae": (
            "Psi-field as universal complex scalar field",
            "matter emerges as modes or excitations",
            "all phenomena as waves or patterns within the field",
            "Psi-field is not an emergent property",
        ),
        "test_protocols": ("classify Psi-field formalisation claims",),
        "null_results": ("Psi-field formalisation requires downstream model tests",),
        "variables": ("Psi", "matter", "modes", "excitations"),
        "validation_targets": (
            "preserve complex-scalar label",
            "preserve matter-emergence direction",
            "preserve source-bound metaphor status",
        ),
        "null_controls": (
            "matter-generates-Psi control must be rejected",
            "metaphor-as-validation control must be rejected",
        ),
    },
    "axiom_i_psi_field.predictive_coding_generative_model": {
        "context_id": "predictive_coding_generative_model",
        "validation_protocol": "paper0.axiom_i_psi_field.predictive_coding_generative_model",
        "canonical_statement": (
            "The Axiom I section maps the Psi-field onto predictive-coding language "
            "as the physical substrate of a cosmic-scale generative model."
        ),
        "source_equation_ids": (
            "P0R00687:predictive_coding_integration",
            "P0R00688:cosmic_generative_model_nature",
            "P0R00689:psi_field_is_generative_model",
            "P0R00690:beliefs_priors_physical_substance",
        ),
        "source_formulae": (
            "cosmic generative model",
            "universe as a system of beliefs or priors",
            "physical substance of cosmic-scale generative model",
            "phenomena as generated content",
        ),
        "test_protocols": ("preserve predictive-coding interpretation boundary",),
        "null_results": ("generative-model mapping is not empirical confirmation",),
        "variables": ("Psi", "priors", "beliefs", "phenomena"),
        "validation_targets": (
            "preserve cosmic-generative-model label",
            "preserve priors-as-source-language boundary",
            "preserve no-measured-prior boundary",
        ),
        "null_controls": (
            "predictive-coding-map-as-data control must be rejected",
            "cosmic-priors-as-fitted-variables control must be rejected",
        ),
    },
    "axiom_i_psi_field.hint_ontological_ground": {
        "context_id": "hint_ontological_ground",
        "validation_protocol": "paper0.axiom_i_psi_field.hint_ontological_ground",
        "canonical_statement": (
            "The source defines the Psi_s term in H_int as ontological ground rather "
            "than a peer interaction field alongside sigma."
        ),
        "source_equation_ids": (
            "P0R00691:psis_field_coupling_integration",
            "P0R00692:H_int=-lambda*Psi_s*sigma",
            "P0R00693:psis_ontological_ground",
            "P0R00694:psis_not_peer_field",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "Psi_s as ontological ground",
            "Psi_s is not one field among many",
            "matter constituting sigma is derived from Psi_s",
        ),
        "test_protocols": ("classify H_int ontological-ground role",),
        "null_results": ("H_int source role is not a fitted coupling result",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma"),
        "validation_targets": (
            "preserve Psi_s not-peer-field boundary",
            "preserve sigma-derived-from-Psi_s boundary",
            "preserve H_int context not-validation boundary",
        ),
        "null_controls": (
            "Psi_s-as-peer-field control must be rejected",
            "H_int-as-hardware-fit control must be rejected",
        ),
    },
    "axiom_i_psi_field.multilayer_consciousness_definition": {
        "context_id": "multilayer_consciousness_definition",
        "validation_protocol": "paper0.axiom_i_psi_field.multilayer_consciousness_definition",
        "canonical_statement": (
            "The source frames consciousness as a multi-layered hierarchy spanning "
            "ontological, physical, and experiential aspects, with an image boundary "
            "kept out of validation evidence."
        ),
        "source_equation_ids": (
            "P0R00695:consciousness_within_scpn_framework",
            "P0R00696:multilayer_hierarchical_model",
            "P0R00697:ontological_physical_experiential_layers",
            "P0R00698:image_boundary",
            "P0R00702:blank_separator",
        ),
        "source_formulae": (
            "consciousness is not a single property",
            "multi-layered hierarchical model",
            "ontological physical experiential aspects",
            "image boundary is not validation evidence",
        ),
        "test_protocols": ("preserve multilayer definition and image boundary",),
        "null_results": ("image boundary contributes no empirical validation",),
        "variables": ("consciousness", "ontology", "physics", "experience"),
        "validation_targets": (
            "preserve multilayer-not-single-property boundary",
            "preserve ontological-physical-experiential labels",
            "preserve image-boundary exclusion",
        ),
        "null_controls": (
            "single-property-collapse control must be rejected",
            "image-as-data control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIPsiFieldSpec:
    """Axiom I Psi-field spec promoted from Paper 0 records."""

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
class AxiomIPsiFieldSpecBundle:
    """Axiom I Psi-field specs plus source coverage summary."""

    specs: tuple[AxiomIPsiFieldSpec, ...]
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


def build_axiom_i_psi_field_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIPsiFieldSpecBundle:
    """Build source-covered Axiom I Psi-field specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIPsiFieldSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIPsiFieldSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0].get("section_path", "")),
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
                implementation_status="implemented_source_accounting_fixture",
                domain_review_status="requires_domain_review_before_public_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Axiom I Psi-Field Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00703",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIPsiFieldSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIPsiFieldSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_i_psi_field_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIPsiFieldSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom I Psi-Field Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Promoted Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae / source labels:",
            ]
        )
        for formula in spec.source_formulae:
            lines.append(f"- {formula}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: AxiomIPsiFieldSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_i_psi_field_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_axiom_i_psi_field_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom I psi-field specs from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
