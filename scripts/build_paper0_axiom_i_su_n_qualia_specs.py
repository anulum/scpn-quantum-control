#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom I SU(N) qualia spec builder
"""Promote Paper 0 Axiom I SU(N) qualia-confinement records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(757, 761))
BLANK_SEPARATOR_IDS = ("P0R00760",)
CLAIM_BOUNDARY = "source-bounded Axiom I SU(N) qualia-confinement map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_i_su_n_qualia.group_extension": {
        "context_id": "group_extension",
        "validation_protocol": "paper0.axiom_i_su_n_qualia.group_extension",
        "canonical_statement": (
            "The source promotes the Abelian U(1) minimal phase framework to an "
            "SU(N) gauge group for N primary qualic dimensions to address "
            "qualitative multiplicity of the qualia fibre space."
        ),
        "source_equation_ids": (
            "P0R00757:u1_minimal_phase_boundary",
            "P0R00757:su_n_gauge_group_extension",
            "P0R00757:info_gluon_count",
        ),
        "source_formulae": (
            "U(1) provides a minimal realizer for intentional phase",
            "U(1) fails qualitative multiplicity of the qualia fiber space",
            "SU(N) gauge group for N primary qualic dimensions",
            "N^2-1 info-gluons",
            "non-linear self-interacting gauge bosons",
        ),
        "test_protocols": ("preserve SU(N) extension source accounting",),
        "null_results": ("SU(N) promotion is a source proposal, not observed gauge evidence",),
        "variables": ("U1", "SU_N", "N", "info_gluon"),
        "validation_targets": (
            "preserve U1 minimality boundary",
            "preserve SU(N) primary-qualic-dimension proposal",
            "preserve N^2-1 info-gluon count formula",
        ),
        "null_controls": (
            "su_n_as_empirical_detection control must be rejected",
            "missing_info_gluon_count control must be rejected",
        ),
    },
    "axiom_i_su_n_qualia.confinement_hypothesis": {
        "context_id": "confinement_hypothesis",
        "validation_protocol": "paper0.axiom_i_su_n_qualia.confinement_hypothesis",
        "canonical_statement": (
            "The source analogises QCD confinement: self-interacting info-gluons "
            "generate Qualia Confinement, with potential energy between "
            "fundamental qualia charges increasing linearly with distance."
        ),
        "source_equation_ids": (
            "P0R00758:confinement_hypothesis",
            "P0R00758:qcd_analogy",
            "P0R00758:linear_potential",
            "P0R00758:holistic_context_explanation",
        ),
        "source_formulae": (
            "Qualia Confinement infrared slavery",
            "self-interaction of info-gluons",
            "V(r) approx sigma r",
            "fundamental qualia charges cannot be isolated from holistic context",
        ),
        "test_protocols": ("preserve qualia-confinement formula boundary",),
        "null_results": ("linear potential is not a fitted string-tension measurement",),
        "variables": ("V", "r", "sigma", "qualia_charge", "info_gluon"),
        "validation_targets": (
            "preserve QCD-analogy boundary",
            "preserve linear confinement potential formula",
            "preserve holistic-context explanation boundary",
        ),
        "null_controls": (
            "linear-potential-as-fit control must be rejected",
            "isolated-qualia-charge control must be rejected",
        ),
    },
    "axiom_i_su_n_qualia.macroscopic_colored_state": {
        "context_id": "macroscopic_colored_state",
        "validation_protocol": "paper0.axiom_i_su_n_qualia.macroscopic_colored_state",
        "canonical_statement": (
            "The source defines the unified Self as a non-trivial macroscopic "
            "coloured state, represented as an irreducible tensor product of "
            "confined qualia charges, and links non-zero net charge to manifold "
            "topology while marking the next Axiom II boundary."
        ),
        "source_equation_ids": (
            "P0R00759:macroscopic_colored_state",
            "P0R00759:irreducible_tensor_product",
            "P0R00759:betti_number_topology",
            "P0R00760:blank_separator",
        ),
        "source_formulae": (
            "Layer 5 unified Self as macroscopic colored state",
            "irreducible tensor product of confined qualia charges",
            "non-zero net charge dictates geometric and topological structure",
            "Betti numbers beta_k of the Consciousness Manifold",
            "blank separator before Axiom II",
        ),
        "test_protocols": ("preserve macroscopic-coloured-state source boundary",),
        "null_results": ("Betti-number link is not a measured topology result in this slice",),
        "variables": ("Layer_5", "beta_k", "qualia_charge", "Consciousness_Manifold"),
        "validation_targets": (
            "preserve Layer 5 coloured-state definition",
            "preserve confined-charge tensor-product statement",
            "preserve Betti-number topology statement",
            "preserve Axiom II boundary",
        ),
        "null_controls": (
            "colored-state-as-measurement control must be rejected",
            "blank-separator-as-content control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomISUNQualiaSpec:
    """Axiom I SU(N) qualia spec promoted from Paper 0 records."""

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
class AxiomISUNQualiaSpecBundle:
    """Axiom I SU(N) qualia specs plus source coverage summary."""

    specs: tuple[AxiomISUNQualiaSpec, ...]
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


def build_axiom_i_su_n_qualia_specs(
    source_records: list[dict[str, Any]],
) -> AxiomISUNQualiaSpecBundle:
    """Build source-covered Axiom I SU(N) qualia specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomISUNQualiaSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomISUNQualiaSpec(
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
        "title": "Paper 0 Axiom I SU(N) Qualia Confinement Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "gauge_boson_formula_count": 1,
        "confinement_formula_count": 1,
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00761",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomISUNQualiaSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomISUNQualiaSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_i_su_n_qualia_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomISUNQualiaSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom I SU(N) Qualia Confinement Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Gauge boson formulae: {bundle.summary['gauge_boson_formula_count']}",
        f"- Confinement formulae: {bundle.summary['confinement_formula_count']}",
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
    bundle: AxiomISUNQualiaSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_i_su_n_qualia_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_axiom_i_su_n_qualia_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
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
