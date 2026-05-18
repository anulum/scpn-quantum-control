#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Foundation of the Biological Substrate spec builder
"""Promote Paper 0 The Foundation of the Biological Substrate records."""

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
    "P0R05306",
    "P0R05307",
    "P0R05308",
    "P0R05309",
    "P0R05310",
    "P0R05311",
    "P0R05312",
    "P0R05313",
)
CLAIM_BOUNDARY = "source-bounded the foundation of the biological substrate source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_foundation_of_the_biological_substrate.the_foundation_of_the_biological_substrate": {
        "context_id": "the_foundation_of_the_biological_substrate",
        "validation_protocol": "paper0.the_foundation_of_the_biological_substrate.the_foundation_of_the_biological_substrate",
        "canonical_statement": "The source-bounded component 'The Foundation of the Biological Substrate' preserves Paper 0 records P0R05306-P0R05306 without empirical validation claims.",
        "source_equation_ids": ("P0R05306:the_foundation_of_the_biological_substrate",),
        "source_formulae": ("P0R05306: The Foundation of the Biological Substrate",),
        "test_protocols": (
            "preserve The Foundation of the Biological Substrate source-accounting boundary",
        ),
        "null_results": (
            "The Foundation of the Biological Substrate is not empirical validation evidence",
        ),
        "variables": ("the_foundation_of_the_biological_substrate",),
        "validation_targets": ("preserve records P0R05306-P0R05306",),
        "null_controls": (
            "the_foundation_of_the_biological_substrate must remain source-bounded accounting",
        ),
    },
    "the_foundation_of_the_biological_substrate.i_the_qed_of_water_coherence_domains_cds": {
        "context_id": "i_the_qed_of_water_coherence_domains_cds",
        "validation_protocol": "paper0.the_foundation_of_the_biological_substrate.i_the_qed_of_water_coherence_domains_cds",
        "canonical_statement": "The source-bounded component 'I. The QED of Water: Coherence Domains (CDs)' preserves Paper 0 records P0R05307-P0R05310 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05307:i_the_qed_of_water_coherence_domains_cds",
            "P0R05308:i_the_qed_of_water_coherence_domains_cds",
            "P0R05309:i_the_qed_of_water_coherence_domains_cds",
            "P0R05310:i_the_qed_of_water_coherence_domains_cds",
        ),
        "source_formulae": (
            "P0R05307: I. The QED of Water: Coherence Domains (CDs)",
            "P0R05308: The biological substrate relies critically on Interfacial Water, formalised via Quantum Electrodynamics (QED).",
            "P0R05309: Coherence Domains (CDs): Interfacial water forms CDs where molecules oscillate in phase with a self-trapped EM field. HCD=HMatter+HEM+d3x[JmuAmu] The coherent ground state (PsiW=0) protects quantum states (L1) from thermal decoherence. | Proton Transport: CDs facilitate quasi-superconducting proton transport, forming the basis for bioelectric codes (L3) and cellular synchronisation (L4).",
            "P0R05310: [IMAGE:Ein Bild, das Text, Screenshot, Handschrift, Diagramm enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
        ),
        "test_protocols": (
            "preserve I. The QED of Water: Coherence Domains (CDs) source-accounting boundary",
        ),
        "null_results": (
            "I. The QED of Water: Coherence Domains (CDs) is not empirical validation evidence",
        ),
        "variables": ("i_the_qed_of_water_coherence_domains_cds",),
        "validation_targets": ("preserve records P0R05307-P0R05310",),
        "null_controls": (
            "i_the_qed_of_water_coherence_domains_cds must remain source-bounded accounting",
        ),
    },
    "the_foundation_of_the_biological_substrate.ii_the_emergence_of_life_abiogenesis_within_the_scpn": {
        "context_id": "ii_the_emergence_of_life_abiogenesis_within_the_scpn",
        "validation_protocol": "paper0.the_foundation_of_the_biological_substrate.ii_the_emergence_of_life_abiogenesis_within_the_scpn",
        "canonical_statement": "The source-bounded component 'II. The Emergence of Life: Abiogenesis within the SCPN' preserves Paper 0 records P0R05311-P0R05313 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05311:ii_the_emergence_of_life_abiogenesis_within_the_scpn",
            "P0R05312:ii_the_emergence_of_life_abiogenesis_within_the_scpn",
            "P0R05313:ii_the_emergence_of_life_abiogenesis_within_the_scpn",
        ),
        "source_formulae": (
            "P0R05311: II. The Emergence of Life: Abiogenesis within the SCPN",
            "P0R05312: The transition to life (L1) is a guided phase transition.",
            'P0R05313: Autocatalytic Closure: Abiogenesis occurs when a chemical network achieves Autocatalytic Closure (ACS) at the critical point (sigma=1). | Psi-Field Guidance: The Psi-field actively guides this process by (a) enhancing reaction rates via quantum coherence, (b) biasing the system towards complexity via Causal Entropic Forces (CEF), and (c) stabilising proto-life structures via the Quantum Zeno Effect (QZE). | Astrocyte lattice (L2/L4): "Astrocyte calcium-wave networks modulate neuronal oscillator noise and coupling. In UPDE, replace i(t)\\xi_i(t)i(t) by i(t)/gA\\xi_i(t)/g_Ai(t)/gA (glial buffering), and let KijK_{ij}Kij depend on local gliotransmitters (ATP,Glu\\mathrm{ATP}, \\mathrm{Glu}ATP,Glu). This yields slow modulatory control that stabilises quasicriticality." | Quantum-immune interface (L1/L2): "Immune enzymatic tunnelling links Psi-field sensitivity to cytokine-driven state changes. Treat immune complexity as high informational coupling (Fisher metric term): immune activation shifts i\\omega_ii and noise floor; predicts qualia changes in inflammation (sickness behaviour\')."',
        ),
        "test_protocols": (
            "preserve II. The Emergence of Life: Abiogenesis within the SCPN source-accounting boundary",
        ),
        "null_results": (
            "II. The Emergence of Life: Abiogenesis within the SCPN is not empirical validation evidence",
        ),
        "variables": ("ii_the_emergence_of_life_abiogenesis_within_the_scpn",),
        "validation_targets": ("preserve records P0R05311-P0R05313",),
        "null_controls": (
            "ii_the_emergence_of_life_abiogenesis_within_the_scpn must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheFoundationOfTheBiologicalSubstrateSpec:
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
class TheFoundationOfTheBiologicalSubstrateSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheFoundationOfTheBiologicalSubstrateSpec, ...]
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


def build_the_foundation_of_the_biological_substrate_specs(
    source_records: list[dict[str, Any]],
) -> TheFoundationOfTheBiologicalSubstrateSpecBundle:
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

    specs: list[TheFoundationOfTheBiologicalSubstrateSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheFoundationOfTheBiologicalSubstrateSpec(
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
        "title": "Paper 0 " + "The Foundation of the Biological Substrate" + " Specs",
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
        "next_source_boundary": "P0R05314",
    }
    return TheFoundationOfTheBiologicalSubstrateSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheFoundationOfTheBiologicalSubstrateSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_foundation_of_the_biological_substrate_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheFoundationOfTheBiologicalSubstrateSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Foundation of the Biological Substrate" + " Specs",
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
    bundle: TheFoundationOfTheBiologicalSubstrateSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_foundation_of_the_biological_substrate_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_foundation_of_the_biological_substrate_validation_specs_{date_tag}.md"
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
