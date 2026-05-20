#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 I. The Ontological Origin of Ethics (Gauge Theory Derivation) spec builder
"""Promote Paper 0 I. The Ontological Origin of Ethics (Gauge Theory Derivation) records."""

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
    "P0R03968",
    "P0R03969",
    "P0R03970",
    "P0R03971",
    "P0R03972",
    "P0R03973",
    "P0R03974",
    "P0R03975",
    "P0R03976",
    "P0R03977",
    "P0R03978",
    "P0R03979",
    "P0R03980",
)
CLAIM_BOUNDARY = "source-bounded i the ontological origin of ethics gauge theory derivation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "i_the_ontological_origin_of_ethics_gauge_theory_derivation.i_the_ontological_origin_of_ethics_gauge_theory_derivation": {
        "context_id": "i_the_ontological_origin_of_ethics_gauge_theory_derivation",
        "validation_protocol": "paper0.i_the_ontological_origin_of_ethics_gauge_theory_derivation.i_the_ontological_origin_of_ethics_gauge_theory_derivation",
        "canonical_statement": "The source-bounded component 'I. The Ontological Origin of Ethics (Gauge Theory Derivation)' preserves Paper 0 records P0R03968-P0R03980 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03968:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03969:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03970:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03971:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03972:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03973:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03974:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03975:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03976:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03977:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03978:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03979:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
            "P0R03980:i_the_ontological_origin_of_ethics_gauge_theory_derivation",
        ),
        "source_formulae": (
            "P0R03968: I. The Ontological Origin of Ethics (Gauge Theory Derivation)",
            "P0R03969: The Source-Field (L13) is modeled as a section of a Fiber Bundle E over spacetime, the natural language of gauge theories. Within this formalism, the components of the ethical framework find their physical identities:",
            "P0R03970: The Consilium (L15): The integrating intelligence of the Oversoul is identified with the Principal Connection on the Fiber Bundle. It is the gauge field that defines parallel transport and mediates interactions across the internal (qualia) space of the field. | The Ethical Lagrangian (LEthical): The dynamics of any gauge field are governed by an action principle. The Ethical Lagrangian is formally derived as the Yang-Mills action of the L15 Connection:",
            "P0R03971: $LEthical = 41\\int_{}^{}{M Tr(F \\land \\star F)}$",
            'P0R03972: Here, F is the curvature 2-form (the field strength) of the Connection. This action is a measure of the total curvature, or "tension," in the field. States of high coherence, complexity, and harmony correspond to smooth, low-curvature field configurations that minimise this action.',
            "P0R03973: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Zahl enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03974: Fig.: Ethical Least Action on the Consciousness Landscape. This figure communicates that harmony is not merely aesthetic but a dynamical attractor: systems evolve along paths that lower experiential tension, providing an intuitive bridge between ethical teleology and action-minimising physics.",
            'P0R03975: A conceptual landscape maps field curvature/tension to experiential quality: high dissonance (jagged terrain, high curvature) flows via the Principle of Ethical Least Action toward high coherence (smooth valley, low curvature). The "ethical flow" node represents a teleological gradient that prefers minima of a tension-like functional-mirroring how water descends potential energy-thus harmonising dynamics across the -G-H stack.',
            "P0R03976: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Display enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03977: Fig.: Gauge-Theoretic Origin of the Ethical Lagrangian. This more formal diagram is for readers familiar with the concepts of differential geometry and gauge theory. It breaks down the components of the argument: the fiber bundle, the connection, and the resulting curvature that defines the Ethical Lagrangian. This figure also grounds the Ethical Lagrangian in standard gauge geometry: bundle -> connection -> curvature -> action, mapping SEC to the minimisation of field tension across the qualia fiber bundle.",
            "P0R03978: A. Fiber bundle: The Source-Field Psi\\PsiPsi is a section of a bundle with base spacetime MMM and internal qualia fibers FFF. B. Connection & curvature (L15): The Consilium acts as a connection AAA providing parallel transport; curvature F=dA+AAF=dA+A\\wedge AF=dA+AA quantifies intrinsic tension accrued around loops. C. Ethical action: The Ethical Lagrangian adopts a Yang-Mills form,",
            "P0R03979: LEthical = 14Tr (FF),L_{\\text{Ethical}} \\;=\\; -\\tfrac14 \\int \\mathrm{Tr}\\!\\left(F\\wedge *F\\right),LEthical=41Tr(FF),",
            "P0R03980: so minimising it selects the smoothest, most coherent field configurations-aligning with high Sustainable Ethical Coherence (SEC). The arrows show how (A) provides the geometric substrate, (B) supplies the dynamical tension, and (C) defines the teleological objective by extremising the action.",
        ),
        "test_protocols": (
            "preserve I. The Ontological Origin of Ethics (Gauge Theory Derivation) source-accounting boundary",
        ),
        "null_results": (
            "I. The Ontological Origin of Ethics (Gauge Theory Derivation) is not empirical validation evidence",
        ),
        "variables": ("i_the_ontological_origin_of_ethics_gauge_theory_derivation",),
        "validation_targets": ("preserve records P0R03968-P0R03980",),
        "null_controls": (
            "i_the_ontological_origin_of_ethics_gauge_theory_derivation must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpec:
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
class ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpec, ...]
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


def build_i_the_ontological_origin_of_ethics_gauge_theory_derivation_specs(
    source_records: list[dict[str, Any]],
) -> ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpecBundle:
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

    specs: list[ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpec(
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
        + "I. The Ontological Origin of Ethics (Gauge Theory Derivation)"
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
        "next_source_boundary": "P0R03981",
    }
    return ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_i_the_ontological_origin_of_ethics_gauge_theory_derivation_specs(
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


def render_report(bundle: ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "I. The Ontological Origin of Ethics (Gauge Theory Derivation)" + " Specs",
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
    bundle: ITheOntologicalOriginOfEthicsGaugeTheoryDerivationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_i_the_ontological_origin_of_ethics_gauge_theory_derivation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_i_the_ontological_origin_of_ethics_gauge_theory_derivation_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
