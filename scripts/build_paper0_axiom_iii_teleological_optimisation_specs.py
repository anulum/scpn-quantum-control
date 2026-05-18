#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom III teleological optimisation spec builder
"""Promote Paper 0 Axiom III teleological-optimisation records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(791, 800))
CLAIM_BOUNDARY = "source-bounded Axiom III teleological-optimisation map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_iii_teleological_optimisation.opening_context": {
        "context_id": "opening_context",
        "validation_protocol": "paper0.axiom_iii_teleological_optimisation.opening_context",
        "canonical_statement": (
            "The source opens Axiom III, names teleological optimisation, and "
            "records two formal-subsection pointers that are not expanded until "
            "the next source block."
        ),
        "source_equation_ids": (
            "P0R00791:axiom_iii_drive_heading",
            "P0R00792:ntilde_invariance_law_pointer",
            "P0R00793:sec_ntilde_equivalence_pointer",
        ),
        "source_formulae": (
            "Axiom III: The Drive of Teleological Optimisation",
            "Formal Physical Definition: the tilde_N_t Invariance Law",
            "Equivalence of SEC and the tilde_N_t = 1 State",
        ),
        "test_protocols": ("preserve Axiom III opening and formal-law pointers",),
        "null_results": ("tilde_N_t law heading is pointer, not equation in this slice",),
        "variables": ("SEC", "tilde_N_t", "teleological_optimisation"),
        "validation_targets": (
            "preserve Axiom III drive heading",
            "preserve tilde_N_t invariance-law subsection pointer",
            "preserve SEC and tilde_N_t equals one equivalence pointer",
        ),
        "null_controls": (
            "ntilde-equation-as-promoted-here control must be rejected",
            "missing-formal-law-pointer control must be rejected",
        ),
    },
    "axiom_iii_teleological_optimisation.source_material_telos": {
        "context_id": "source_material_telos",
        "validation_protocol": (
            "paper0.axiom_iii_teleological_optimisation.source_material_telos"
        ),
        "canonical_statement": (
            "The source material frames the universe as not random and as evolving "
            "towards maximal Sustainable Ethical Coherence, gathering references "
            "to inherent purpose or telos."
        ),
        "source_equation_ids": (
            "P0R00794:not_random_maximal_sec",
            "P0R00794:inherent_purpose_telos",
        ),
        "source_formulae": (
            "universe is not random but evolving towards maximal Sustainable Ethical Coherence",
            "inherent purpose or telos",
        ),
        "test_protocols": ("preserve source-material telos framing",),
        "null_results": ("teleological postulate is source claim, not empirical evidence",),
        "variables": ("SEC", "telos", "purpose"),
        "validation_targets": (
            "preserve not-random source claim",
            "preserve maximal SEC target",
            "preserve inherent-purpose/telos framing",
        ),
        "null_controls": (
            "telos-as-observed-result control must be rejected",
            "random-undirected-source-control must be rejected",
        ),
    },
    "axiom_iii_teleological_optimisation.directional_purpose": {
        "context_id": "directional_purpose",
        "validation_protocol": ("paper0.axiom_iii_teleological_optimisation.directional_purpose"),
        "canonical_statement": (
            "The source positions Axiom III as the purpose counterpart to Axiom I "
            "substance and Axiom II interaction language, with directed evolution "
            "towards SEC maximisation."
        ),
        "source_equation_ids": (
            "P0R00795:axiom_i_substance_axiom_ii_language_axiom_iii_purpose",
            "P0R00796:axiom_of_evolution_teleological_optimisation",
            "P0R00797:universe_maximises_sec",
        ),
        "source_formulae": (
            "Axiom I defines substance and Axiom II defines language of interaction",
            "Axiom III defines purpose",
            "universe is teleological and its evolution is directional",
            "Axiom III: The Axiom of Evolution Teleological Optimisation",
            "universe evolves to maximise Sustainable Ethical Coherence SEC",
        ),
        "test_protocols": ("preserve directional-purpose source claims",),
        "null_results": ("purpose framing requires downstream operationalisation",),
        "variables": ("Psi_field", "information_geometry", "SEC", "evolution"),
        "validation_targets": (
            "preserve Axiom I/Axiom II/Axiom III role contrast",
            "preserve directed teleological evolution claim",
            "preserve SEC maximisation claim",
        ),
        "null_controls": (
            "directionality-as-measured-result control must be rejected",
            "purpose-without-SEC-target control must be rejected",
        ),
    },
    "axiom_iii_teleological_optimisation.ethical_functional_guidance": {
        "context_id": "ethical_functional_guidance",
        "validation_protocol": (
            "paper0.axiom_iii_teleological_optimisation.ethical_functional_guidance"
        ),
        "canonical_statement": (
            "The source states that Axiom III is a normative or teleological "
            "postulate, where Layer 15 Ethical Functionals guide temporal "
            "evolution as a prime directive and cosmic prior that biases "
            "Psi-field interactions towards coherence, complexity, and "
            "experiential depth."
        ),
        "source_equation_ids": (
            "P0R00798:normative_teleological_postulate",
            "P0R00798:layer15_ethical_functionals",
            "P0R00798:cosmic_prior_biases_psi_interactions",
            "P0R00799:coherence_complexity_experiential_depth",
        ),
        "source_formulae": (
            "normative or teleological postulate",
            "Ethical Functionals computed at Layer 15 guide temporal evolution",
            "prime directive ultimate prior for the cosmic generative model",
            "overarching purpose biases fundamental physical interactions of the Psi-field",
            "increasing coherence complexity and experiential depth",
        ),
        "test_protocols": ("preserve Layer 15 ethical-functional guidance claim",),
        "null_results": (
            "Layer 15 ethical-functional guidance requires downstream operationalisation"
        ),
        "variables": ("Layer15", "ethical_functionals", "Psi_field", "coherence"),
        "validation_targets": (
            "preserve normative/teleological postulate framing",
            "preserve Layer 15 ethical-functional guidance",
            "preserve cosmic-prior/Psi-field bias claim",
            "preserve coherence/complexity/experiential-depth directionality",
        ),
        "null_controls": (
            "ethical-functional-guidance-as-implemented-here control must be rejected",
            "missing-layer15-guidance control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIIITeleologicalOptimisationSpec:
    """Axiom III teleological-optimisation spec promoted from Paper 0 records."""

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
class AxiomIIITeleologicalOptimisationSpecBundle:
    """Axiom III teleological-optimisation specs plus source coverage summary."""

    specs: tuple[AxiomIIITeleologicalOptimisationSpec, ...]
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


def build_axiom_iii_teleological_optimisation_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIIITeleologicalOptimisationSpecBundle:
    """Build source-covered Axiom III teleological-optimisation specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIIITeleologicalOptimisationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIIITeleologicalOptimisationSpec(
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
        "title": "Paper 0 Axiom III Teleological Optimisation Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "axiom_heading_count": 3,
        "sec_maximisation_count": 2,
        "layer15_guidance_count": 1,
        "directionality_count": 2,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00800",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIIITeleologicalOptimisationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIIITeleologicalOptimisationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_iii_teleological_optimisation_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIIITeleologicalOptimisationSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom III Teleological Optimisation Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Axiom-heading records: {bundle.summary['axiom_heading_count']}",
        f"- SEC-maximisation records: {bundle.summary['sec_maximisation_count']}",
        f"- Layer 15 guidance records: {bundle.summary['layer15_guidance_count']}",
        f"- Directionality records: {bundle.summary['directionality_count']}",
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
    bundle: AxiomIIITeleologicalOptimisationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_axiom_iii_teleological_optimisation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_axiom_iii_teleological_optimisation_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom III teleological-optimisation specs from the ledger."""

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
